from re import finditer
import numpy as np
import sounddevice as sd
import soundfile as sf
import pretty_midi

# Create a PrettyMIDI object
cello_c_chord = pretty_midi.PrettyMIDI()
# Create an Instrument instance for a cello instrument
cello_program = pretty_midi.instrument_name_to_program('Cello')
cello = pretty_midi.Instrument(program=cello_program)
# Iterate over note names, which will be converted to note number later
for i, note_name in enumerate(['C5', 'E5']):
    # Retrieve the MIDI note number for this note name
    note_number = pretty_midi.note_name_to_number(note_name)
    # Create a Note instance for this note, starting at 0s and ending at .5s
    note = pretty_midi.Note(velocity=1, pitch=note_number, start=i, end=i+1)
    # Add it to our cello instrument
    cello.notes.append(note)
# Add the cello instrument to the PrettyMIDI object
cello_c_chord.instruments.append(cello)
# Write out the MIDI data
audio_data = cello_c_chord.synthesize()

# Frames = audio length * sample rate

data, fs = sf.read('samples/sample1.wav', frames=len(audio_data))
data = data[:, 0] # TODO: convert to mono
sd.play(audio_data, fs)

fitness = (data - audio_data)**2