from re import finditer
import numpy as np
import sounddevice as sd
import soundfile as sf
import pretty_midi



chord = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
instrument.notes = [pretty_midi.Note(velocity=100, pitch=pitch, start=0, end=.5) for pitch in [60, 67, 64]]
chord.instruments = [instrument]
# Write out the MIDI data
audio_data = chord.fluidsynth()


midi_data = pretty_midi.PrettyMIDI('samples/scale_c_major.mid')
audio_data = midi_data.fluidsynth()

# Frames = audio length * sample rate
data, fs = sf.read('samples/scale_c_major.wav', frames=len(audio_data))
data = data # TODO: convert to mono
sd.play(audio_data, fs)
