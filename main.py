from re import finditer
import numpy as np
import sounddevice as sd
import soundfile as sf
import pretty_midi
import itertools
import inspyred
from random import Random


MIN_PITCH, MAX_PITCH = 21, 108

target, fs = sf.read('samples/scale_c_major.wav', frames=int(4.5*44100))
target_midi = pretty_midi.PrettyMIDI('samples/scale_c_major.mid')
print(f'Target: {[n.pitch for n in target_midi.instruments[0].notes]}')

#audio_data = midi_data.fluidsynth()

chord = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
instrument.notes = [pretty_midi.Note(velocity=100, pitch=pitch, start=0, end=.5) for pitch in range(0,2000)]
chord.instruments = [instrument]
# Write out the MIDI data
audio_data = chord.fluidsynth()


def to_midi(candidate):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
    midi.instruments.append(instrument)
    for i, pitch in enumerate(candidate):
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=int(pitch), start=i*.5, end=i*.5+.5))
    return midi

def compute_fitness(candidate, target_audio):
    candidate_audio = to_midi(candidate).fluidsynth()
    return np.sum((candidate_audio - target_audio) ** 2)

def generator(random, args):
    size = args.get('num_notes')
    return [random.uniform(MIN_PITCH, MAX_PITCH) for i in range(size)]

def bounder(candidate, args):
    size = args.get('num_notes')
    for i, pitch in enumerate(candidate[:size]):
        candidate[i] = min(max(pitch, MIN_PITCH), MAX_PITCH)
    return candidate

@inspyred.ec.evaluators.evaluator
def evaluator(candidate, args):
    size = args.get('num_notes')
    return compute_fitness(candidate[:size], target)
   

def observer(population, num_generations, num_evaluations, args):
    print(f'{num_generations}) best: {population[0].fitness}  {np.array(population[0].candidate, dtype=int)}')

rand = Random()
rand.seed(42)
es = inspyred.ec.ES(rand)
es.variator = [inspyred.ec.variators.uniform_crossover, inspyred.ec.variators.gaussian_mutation]
es.observer = observer
es.terminator = [inspyred.ec.terminators.evaluation_termination, inspyred.ec.terminators.diversity_termination]
final_pop = es.evolve(
    generator=generator, 
    evaluator=evaluator,
    pop_size=100, 
    bounder=bounder,
    max_evaluations=3000,
    maximize=False,
    num_notes=7
)






# Frames = audio length * sample rate
sd.play(audio_data, fs)
