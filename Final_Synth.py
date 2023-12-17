import numpy as np
import pyaudio
import pygame
from pygame import midi
import math
import itertools
import matplotlib.pyplot as plt

params = {"voice_quality_factor": 1.0, "velocity": 0}  # Default factor

def get_sin_oscillator(freq=55, amp=1, sample_rate=44100):
    increment = (2 * math.pi * freq)/ sample_rate
    return (math.sin(v) * amp for v in itertools.count(start=0, step=increment))

def map_mod_wheel_to_factor(mod_wheel_value):
    # Map the mod wheel value (0-127) to a suitable range for the voice quality factor
    return mod_wheel_value / 127.0

# calculate the Oq & Sq from VQ_factor
def calculate_open_quotient(voice_quality_factor):
    # Higher voice quality factor leads to a higher open quotient
    # Open_quo range is 0.4 to 0.8
    Open_quo = 0.4 + 0.4 * voice_quality_factor
    return Open_quo
def calculate_speed_quotient(voice_quality_factor):
    # Higher voice quality factor might correspond to a faster closing phase
    # Speed_quo range is 1.5 to 4.5
    # to-do
    Speed_quo = 3.83-2.83 * voice_quality_factor
    return Speed_quo

def get_white_noise(scale=1, num_samples=64):
    """Generate white noise."""
    return np.random.normal(0, scale, num_samples)

def get_rosenberg_oscillator(freq=55, amp=1, sample_rate=44100, params={"voice_quality_factor": 1.0, "velocity": 0}):
    """
    Generator function to create a Rosenberg glottal pulse waveform.
    """
    T0_s = 1.0 / freq
    voice_quality_factor = params["voice_quality_factor"]

    # TP_RATIO = 0.4  # Ratio of the open phase of the glottal cycle
    # TN_RATIO = 0.16  # Ratio of the return phase of the glottal cycle
    Open_quo = calculate_open_quotient(voice_quality_factor)
    Speed_quo = calculate_speed_quotient(voice_quality_factor)

    TN_RATIO = Open_quo/(1+Speed_quo)
    TP_RATIO = TN_RATIO * Speed_quo

    print("Open quo:", round(Open_quo, 3), "Speed_quo:", round(Speed_quo, 3))
    print("TN_ratio:", round(TN_RATIO, 3), "TP_ratio:", round(TP_RATIO, 3))

    TP_s = TP_RATIO * T0_s
    TN_s = TN_RATIO * T0_s
    T = int(sample_rate / freq)  # Total samples per period


    velocity = params["velocity"]
    # by default, if we don't move the mod wheel, the below VQ_factor = 1
    noise_amp = 0.8 * (velocity / 127) * voice_quality_factor

    noise_scale = noise_amp/3
    print("noise amp (AH):", round(noise_amp,3))
    print("noise scale (AH/3):", round(noise_scale,3))

    prev_combined_output = 0
    while True:
        noise = get_white_noise(scale=noise_scale, num_samples=T)
        for n in range(T):
            t_cycle_s = (n / sample_rate) % T0_s
            if t_cycle_s <= TP_RATIO * T0_s:  # Open phase
                waveform_sample = 0.5 * amp * (1-np.cos(np.pi * t_cycle_s / TP_s))
            elif TP_RATIO * T0_s < t_cycle_s <= (TP_RATIO + TN_RATIO) * T0_s:  # Return phase
                waveform_sample = amp * np.cos(np.pi * (t_cycle_s - TP_s) / (2*TN_s))
            else:  # Closed phase
                waveform_sample = 0
            # yield np.diff(waveform_sample * noise[n] + waveform_sample)
            combined_output = waveform_sample + (waveform_sample * noise[n] * voice_quality_factor)

            derivative = combined_output - prev_combined_output
            prev_combined_output = combined_output  # Update previous output for next iteration

            yield derivative


def fade_in(signal, fade_length):
    fade_in_window = (1 - np.cos(np.linspace(0, np.pi, fade_length))) * 0.5
    signal[:fade_length] = np.multiply(fade_in_window, signal[:fade_length])
    return signal

# inverse of fade_in
def fade_out(signal, fade_length):
    fade_out_window = (1 + np.cos(np.linspace(0, np.pi, fade_length))) * 0.5
    # fade out length is the last num_sample
    signal[- fade_length:] = np.multiply(fade_out_window, signal[- fade_length:])
    return signal



class PolySynth:
    def __init__(self, amp_scale=0.0001, max_amp=0.8, sample_rate=44100, num_samples=64):
        # Initialize MIDI
        midi.init()
        if midi.get_count() > 0:
            self.midi_input = midi.Input(midi.get_default_input_id())
        else:
            raise Exception("no midi devices detected")

        # Constants
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.amp_scale = amp_scale
        self.max_amp = max_amp
        self.notes_dict = {}

    def _init_stream(self):
        # Initialize the Stream object
        self.stream = pyaudio.PyAudio().open(
            rate=self.sample_rate,
            channels=2, # left and right
            format=pyaudio.paInt16,
            output=True,
            frames_per_buffer=self.num_samples
        )

    def plot_waveform(self, waveform, title):
        plt.figure(figsize=(10, 4))
        plt.plot(waveform, label='Waveform')
        plt.title(title)
        plt.xlabel('Sample Number')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()


    def note_off(self, note):
        osc, first_click, sample_history, _ = self.notes_dict[note]
        # Calculate fade length (ensure it's not longer than the sample history)
        fade_length = self.num_samples
        # Apply fade-out to the last samples
        last_samples = sample_history[-fade_length:]
        faded_out_samples = fade_out(np.array(last_samples), fade_length)
        # Replace the last samples in the history with the faded-out samples
        sample_history[-fade_length:] = faded_out_samples.tolist()
        # Plot the final waveform or just plot the final few num_samples
        # self.plot_waveform(sample_history[-20*fade_length:], f'Final Waveform for Note {note}')
        del self.notes_dict[note]


    def _get_samples(self, notes_dict):
        # Generate samples for both channels
        samples = np.zeros((self.num_samples, 2))  # Two columns for stereo

        for note, (osc, first_click, sample_history, remain_duration) in notes_dict.items():
            note_samples = np.array([next(osc) for _ in range(self.num_samples)]) * self.amp_scale

            # fade in
            # if note gets hit the first time, fade in is used
            if first_click:
                #  fade_in length is the first num_sample (64 frames per buffer)
                fade_length = self.num_samples
                first_samples = note_samples[:fade_length]
                note_samples = fade_in(np.array(first_samples), fade_length)
                # set the boolean back to False
                notes_dict[note] = (osc, False, sample_history, remain_duration)

            # # fade out
            # # check if arriving the last buffer of samples for the note.
            # if remain_duration <= self.num_samples:
            #     fade_length = self.num_samples
            #     last_samples = note_samples[-fade_length:]
            #     note_samples = fade_out(np.array(last_samples), fade_length)
            #     # note_samples = fade_out(note_samples, fade_length)
            #     # after fade_out, no more remaining duration, finished playing.
            #     notes_dict[note] = (osc, True, sample_history, 0)
            # else:
            #     # Otherwise, decrement/count down the note_duration to the final num_sample
            #     notes_dict[note] = (osc, first_click, sample_history, remain_duration - self.num_samples)

            sample_history.extend(note_samples.tolist())

            for i in range(self.num_samples):
                if note <= 59:  # Left channel
                    samples[i, 0] += note_samples[i]
                else:  # Right channel
                    samples[i, 1] += note_samples[i]

        # Clip to the range of int16 to avoid overflow
        # samples = np.clip(samples, -self.max_amp, self.max_amp)
        if np.any(np.abs(samples) >= 1):
            print("Warning: Overflow detected.")

        # Convert to int16 format
        samples = np.int16(samples * 32767)
        if np.any(np.max(np.abs(samples)) == 32767):
            print("Warning: Audio clipping detected.")
        return samples


    def plot_note_samples(self, original_samples1, faded_in_samples, original_samples2,faded_out_samples, note_number):
        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig, (ax1, ax3) = plt.subplots(1, 2)

        ax1.plot(original_samples1, color='r', label= 'Original')
        ax1.plot(faded_in_samples, color='g', linestyle='--',label= "Faded in")
        ax1.set_title(f'Original vs FadeIn for MIDI Note {note_number}')
        ax1.set_xlabel('Sample Number')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        ax1.set_xticks(np.linspace(0, 320, 5))
        ax1.legend()

        ax3.plot(original_samples2, color='r', label="Original")
        ax3.plot(faded_out_samples, color='g', linestyle='--',label="Faded Out")
        ax3.set_title(f'Original vs FadeOut for MIDI Note {note_number}')
        ax3.set_xlabel('Sample Number')
        ax3.set_ylabel('Amplitude')
        ax3.grid(True)
        ax3.set_xticks(np.linspace(0, 320, 5))
        ax3.legend()

        plt.tight_layout()
        plt.show()

    def play(self, osc_function=get_rosenberg_oscillator, close=False):
        self._init_stream()

        print("Starting...")
        try:
            # notes_dict = {}

            while True:
                if self.notes_dict:
                    # Play the notes
                    samples = self._get_samples(self.notes_dict)
                    self.stream.write(samples.tobytes())

                if self.midi_input.poll():
                    for event in self.midi_input.read(num_events=16):
                        (status, note, vel, _), _ = event
                        print(event)
                        # print("the velocity",event[1])

                        if status == 0xB0 and note == 1:
                            # Map mod wheel value (0-127) to desired range for voice quality factor
                            # VQ_factor: range from 0 to 1
                            params["voice_quality_factor"] = map_mod_wheel_to_factor(vel)
                            print("VQ_factor (vel/127):", round(params["voice_quality_factor"],3))

                        if status == 0x80 and note in self.notes_dict:  # Note OFF
                            # by calling the note_off function, here it performs the fade out
                            self.note_off(note)
                            # osc, first_click, sample_history, duration = self.notes_dict[note]

                            # plot fade_in
                            # for the plot just take the first 1 num_sample of the history
                            # original_samples1 = sample_history[:10 * self.num_samples]
                            # faded_in_samples = fade_in(np.array(original_samples1), self.num_samples)

                            # self.notes_dict[note] = (osc, first_click, sample_history, self.num_samples)
                            # # plot fade_out, take the last 4 num_sample of the history
                            # original_samples2 = sample_history[-10 * self.num_samples:]
                            # faded_out_samples = fade_out(np.array(original_samples2), self.num_samples)
                            # self.plot_note_samples(original_samples1, faded_in_samples, original_samples2, faded_out_samples, note)
                            # self.plot_note_samples(original_samples1, original_samples1, original_samples2, original_samples2, note)
                            # del self.notes_dict[note]

                        elif status == 0x90:  # Note ON
                            params["velocity"] = vel

                            if note <= 59:
                                freq = midi.midi_to_frequency(note)
                                self.notes_dict[note] = [
                                    get_rosenberg_oscillator(freq=freq, amp=vel / 127, sample_rate=self.sample_rate, params=params),
                                    True,
                                    [],
                                    self.sample_rate
                                ]
                            elif 59 < note <= 84:
                                freq = midi.midi_to_frequency(note - 24)
                                # freq = midi.midi_to_frequency(note)
                                self.notes_dict[note] = [
                                    get_rosenberg_oscillator(freq=freq, amp=vel / 127, sample_rate=self.sample_rate, params=params),
                                    True,
                                    [],
                                    self.sample_rate
                                ]


        except KeyboardInterrupt as err:
            self.stream.close()
            if close:
                self.midi_input.close()
            print("Stopping...")


synth = PolySynth(amp_scale=3.8, num_samples=64)
# synth = PolySynth(amp_scale=0.001,num_samples=44100)
# synth.play(osc_function=get_sin_oscillator)
synth.play(osc_function = get_rosenberg_oscillator)

