from re import finditer
import numpy as np
import sounddevice as sd
import soundfile as sf
import pretty_midi
import itertools
import inspyred
from random import Random
from time import time


MIN_PITCH, MAX_PITCH = 21, 108

def to_midi(candidate):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
    midi.instruments.append(instrument)
    for i, pitch in enumerate(candidate):
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=int(pitch), start=i*.5, end=i*.5+.5))
    return midi

def compute_fitness(candidate, target_audio):
    WND = 4096
    candidate_audio = to_midi(candidate).fluidsynth()
    
    pad_size = WND - (candidate_audio.size %  WND)
    candidate_fft = np.abs(np.fft.rfft(np.pad(candidate_audio, (0, pad_size)).reshape((-1, WND)) * np.hanning(WND)))
    target_fft = np.abs(np.fft.rfft(np.pad(target_audio, (0, pad_size)).reshape((-1, WND)) * np.hanning(WND)))
    fitness = np.maximum(candidate_fft[:, 2:] / target_fft[:, 2:], target_fft[:, 2:] / candidate_fft[:, 2:]).sum()
    # fitness = np.sum((candidate_audio - target_audio) ** 2)
    return fitness

def generator(random, args):
    size = args.get('num_notes')
    return [random.uniform(MIN_PITCH, MAX_PITCH) for i in range(size)]

def bounder(candidate, args):
    size = args.get('num_notes')
    for i, pitch in enumerate(candidate[:size]):
        candidate[i] = min(max(pitch, MIN_PITCH), MAX_PITCH)
    return candidate

def evaluator(candidates, args):
    size = args.get('num_notes')
    target = args.get('audio_target')
    fitnesses = []
    for candidate in candidates:
        fitnesses.append(compute_fitness(candidate[:size], target))
    return fitnesses
   

def observer(population, num_generations, num_evaluations, args):
    print(f'{num_generations}) best: {population[0].fitness:.2f}  {np.array(population[0].candidate, dtype=int)}')
    if num_generations % 2 == 0:
        print('Reproducing best individual...')
        sd.play(to_midi(population[0].candidate).fluidsynth(), fs)



if __name__ == '__main__':
    target, fs = sf.read('samples/scale_c_major.wav', frames=int(4.5*44100))
    target_midi = pretty_midi.PrettyMIDI('samples/scale_c_major.mid')
    print(f'Target: {[n.pitch for n in target_midi.instruments[0].notes]}')

    chord = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
    instrument.notes = [pretty_midi.Note(velocity=100, pitch=pitch, start=0, end=.5) for pitch in range(0,2000)]
    chord.instruments = [instrument]
    # Write out the MIDI data
    audio_data = chord.fluidsynth()

    rand = Random()
    rand.seed(time())
    es = inspyred.ec.GA(rand)
    es.variator = [inspyred.ec.variators.uniform_crossover, inspyred.ec.variators.gaussian_mutation]
    es.observer = observer
    es.terminator = [inspyred.ec.terminators.evaluation_termination, inspyred.ec.terminators.diversity_termination]
    final_pop = es.evolve(
        generator=generator, 
        mp_evaluator=evaluator,
        evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
        mp_num_cpus=4,
        pop_size=100, 
        num_elites=2,
        bounder=bounder,
        max_evaluations=30000,
        maximize=False,
        num_notes=7,
        audio_target=target
    )

    sd.play(audio_data, fs)
