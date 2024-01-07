# Realistic excitation of 3D-printed vocal tracts using a MIDI controller
## Project Overview
This project aimed to achieve simultaneous and polyphonic excitation of 3D-printed vocal tracts. The initial phase involved developing a synthesizer capable of generating real-time excitation signals from a MIDI keyboard. Subsequently, various acoustic features were implemented, leveraging MIDI keyboard controls. These features encompass fade in and fade out effects, white noise generation, and voice quality modulation. 
The primary focus involved the implementation of these functionalities within a Python script (Synthesizer.py). This report provides a comprehensive overview of the synthesizer's implementation, covering hardware setup, and an in-depth analysis of acoustic features. My goal of this project is the generation of correct excitation signals and the addition of various acoustic features.


## Hardware SetUp
<img width="382" alt="image" src="https://github.com/elsachen0907/MIDI-Synthesizer/assets/74115079/afdb8d40-2c5f-425c-9264-dfe2ee163f5d">

### Input
The project utilizes the Korg microKEY 49 as its input device. This MIDI (Musical Instrument Digital Interface) keyboard features 49 keys and serves as a crucial component for the project. MIDI is a communication protocol enabling interaction between computers, musical instruments, and various hardware. Positioned on the left side of the keyboard are a Pitch bend wheel, a Modulation (MOD) wheel, and an Octave shift button. Additionally, the device is powered through USB, and I utilized the USB connection to link it to my laptop, supplying the necessary power.
![image](https://github.com/elsachen0907/MIDI-Synthesizer/assets/74115079/a21c61ec-bce7-42d2-888a-b98c333a963b)

### Synthesizer
The Synthesizer Python script encompasses the entire flow logic. Upon the user's keypress on the MIDI keyboard, the script receives the MIDI signal, processes the received data, and transforms it into the final excitation signal, which is then directed to the output stream.

### Output
Consequently, after the Synthesizer Python program processes the MIDI input, the two output streams will be driven accordingly. Each is connected to an amplifier. Here, in my project, the VISATON AMP 2.2 LN stereo amplifier is being utilized. It allows direct connection to the headphone output of a computer sound card. The amplifier also features a volume adjustment wheel for regulating the sound intensity. On each side of the stereo amplifier, there are two 3D-printed cubic shaping holders, housing these two amplifiers and serve as platforms for attaching the 3D-printed vocal tracts.
![image](https://github.com/elsachen0907/MIDI-Synthesizer/assets/74115079/a1b04c8e-4b27-4a9e-b56f-95c5175548ef)

## Rosenberg Model
The Rosenberg model is a continuous model for describing the vibrations of the vocal cords. It was developed in 1971 by the American linguist Aaron E. Rosenberg. He concluded that the glottal volume velocity pulses have only one slope discontinuity. There are two types of Rosenberg glottal volume velocity pulse models: polynomial and trigonometric.

## Features
