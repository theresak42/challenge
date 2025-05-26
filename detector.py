#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detects onsets, beats and tempo in WAV files.

For usage information, call with --help.

Author: Jan Schlüter
"""

import sys
from pathlib import Path
from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, find_peaks_cwt
from utils import smoothing, salience, salience1, choose_best_agent, remove_duplicate_agents, salience_gaussian, salience_interval_based
from utils import beat_agent, IOICluster

import librosa
try:
    import tqdm
except ImportError:
    tqdm = None


def opts_parser():
    usage =\
"""Detects onsets, beats and tempo in WAV files.
"""
    parser = ArgumentParser(description=usage)
    parser.add_argument('indir',
            type=str,
            help='Directory of WAV files to process.')
    parser.add_argument('outfile',
            type=str,
            help='Output JSON file to write.')
    parser.add_argument('--plot',
            action='store_true',
            help='If given, plot something for every file processed.')
    return parser


def detect_everything(filename, options):
    """
    Computes some shared features and calls the onset, tempo and beat detectors.
    """
    # read wave file (this is faster than librosa.load)
    sample_rate, signal = wavfile.read(filename)

    # convert from integer to float
    if signal.dtype.kind == 'i':
        signal = signal / np.iinfo(signal.dtype).max

    # convert from stereo to mono (just in case)
    if signal.ndim == 2:
        signal = signal.mean(axis=-1)

    # compute spectrogram with given number of frames per second
    fps = 70
    hop_length = sample_rate // fps
    spect = librosa.stft(
            signal, n_fft=2048, hop_length=hop_length, window='hann')

    # only keep the magnitude
    magspect = np.abs(spect)

    # compute a mel spectrogram
    melspect = librosa.feature.melspectrogram(
            S=magspect, sr=sample_rate, n_mels=80, fmin=27.5, fmax=8000)

    # compress magnitudes logarithmically
    melspect = np.log1p(100 * melspect) 

    # compute onset detection function
    odf, odf_rate = onset_detection_function(
            sample_rate, signal, fps, spect, magspect, melspect, options)

    # detect onsets from the onset detection function
    onsets = detect_onsets(odf_rate, odf, options)

    # detect tempo from everything we have
    tempo = detect_tempo(
            sample_rate, signal, fps, spect, magspect, melspect,
            odf_rate, odf, onsets, options)

    # detect beats from everything we have (including the tempo)
    beats = detect_beats(
            sample_rate, signal, fps, spect, magspect, melspect,
            odf_rate, odf, onsets, tempo, options)

    # plot some things for easier debugging, if asked for it
    if options.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, sharex=True)
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle(filename)
        axes[0].set_title('melspect')
        axes[0].imshow(melspect, origin='lower', aspect='auto',
                       extent=(0, melspect.shape[1] / fps,
                               -0.5, melspect.shape[0] - 0.5))
        axes[1].set_title('onsets')
        axes[1].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in onsets:
            axes[1].axvline(position, color='tab:orange')
        axes[2].set_title('beats (tempo: %r)' % list(np.round(tempo, 2)))
        axes[2].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in beats:
            axes[2].axvline(position, color='tab:red')
        plt.show()

    return {'onsets': list(np.round(onsets, 3)),
            'beats': list(np.round(beats, 3)),
            'tempo': list(np.round(tempo, 2))}



    


def onset_detection_function(sample_rate, signal, fps, spect, magspect,
                             melspect, options):
    """
    Compute an onset detection function. Ideally, this would have peaks
    where the onsets are. Returns the function values and its sample/frame
    rate in values per second as a tuple: (values, values_per_second)
    """
    # we only have a dumb dummy implementation here.
    # it returns every 1000th absolute sample value of the input signal.
    # this is not a useful solution at all, just a placeholder.
    values = np.abs(signal[::1000])
    values_per_second = sample_rate / 1000

    h = 1
    x_t = np.log10(1+10*magspect)
    #print(x_t.shape)
    sd_t = x_t[:, h:] - x_t[:, :int(x_t.shape[1])-h]
    #print(sd_t.shape)
    hw_t = np.maximum(0,sd_t)**2
    #print(hw_t.shape)
    dSD_t = np.sum(hw_t, axis=0)

    values_per_second = fps

    #print(dSD_t.shape)
    

    

    return dSD_t, values_per_second


def detect_onsets(odf_rate, odf, options):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """
    # we only have a dumb dummy implementation here.
    # it returns the timestamps of the 100 strongest values.
    # this is not a useful solution at all, just a placeholder.
    #strongest_indices = np.argpartition(odf, 100)[:100]
    #strongest_indices.sort()

    norm_odf = (odf - np.min(odf)) / (np.max(odf) - np.min(odf))
    #print(type(norm_odf), norm_odf.shape)
    norm_odf_smooth = smoothing(norm_odf, window=5)
    #print(type(norm_odf_smooth), norm_odf_smooth.shape)
    #TODO: use smoothing function/ DC removal
    t = "F"
    if t=="T":
        plt.figure(figsize=(10, 4))
        plt.plot(norm_odf_smooth, color='dodgerblue', linewidth=2)
        plt.title("Onset Strength Over Time")
        plt.xlabel("Time Frame (t)")
        plt.ylabel("Strength")
        plt.grid(True)
        plt.tight_layout()
        #plt.savefig("pic.png")
        plt.show()
    #plt.close()

    #input()

    #implement custom peak picking

    threshold = 0.05
    strongest_indices = []
    for i, (prev, curr, nex) in enumerate(zip(norm_odf_smooth[:len(norm_odf-2)],norm_odf_smooth[1:len(norm_odf_smooth-1)],norm_odf[2:])):
        if curr > prev and curr > nex and curr > threshold:
            strongest_indices.append(i+1)

    
    #strongest_indices = find_peaks(norm_odf_smooth, distance = 10)[0]
    strongest_indices = find_peaks_cwt(norm_odf, widths=1)
    #print(norm_odf_smooth)

    #strongest_indices += 1
    #print(strongest_indices)
    
    #print(strongest_indices/odf_rate, len(strongest_indices))
    #input()


    return strongest_indices / odf_rate


def detect_tempo(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, options):
    """
    Detect tempo using any of the input representations.
    Returns one tempo or two tempo estimations.
    """    
    # we only have a dumb dummy implementation here.
    # it uses the time difference between the first two onsets to
    # define the tempo, and returns half of that as a second guess.
    # this is not a useful solution at all, just a placeholder.
    #tempo = 60 / (onsets[1] - onsets[0])
    #return [tempo / 2, tempo]
    clusters = IOICluster()
    tempo = clusters.run(onsets)
    return tempo


def detect_beats(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, tempo, options):
    """
    Detect beats using any of the input representations.
    Returns the positions of all beats in seconds.
    """
    # we only have a dumb dummy implementation here.
    # it returns every 10th onset as a beat.
    # this is not a useful solution at all, just a placeholder.

    print("Onsets: ")
    print(onsets[:20])
    print(len(onsets))

    print("Tempo:")
    print(tempo[0])
    startup_period = len(onsets)//10
    timeout = 5
    correction_factor = 10
    tol_inner = 40./1000. # s = 40 ms
    p_tol_pre = 0.2
    p_tol_post= 0.4

    #return onsets[::10]


    """    clusters = IOICluster()
        tempo = clusters.run(onsets)
        print(f"My tempi: {tempo}")
    """
    # Initialization
    agents = []
    for t_i in tempo:
        if t_i > 0:
            for e_j in onsets[:startup_period]:
                agent = beat_agent(beat_interval=60/t_i, 
                                pred = e_j+60/t_i, 
                                hist = [e_j], 
                                score = salience_interval_based(60./t_i))
                agents.append(agent)
    print(f"Initialized {len(agents)} agents.")

    # Main loop
    for _, e_j in enumerate(onsets):
        for i, a_i in enumerate(agents):
            new_agents = []
            to_be_deleted = []
            if e_j - a_i.history[-1] > timeout:
                to_be_deleted.append(i)
                continue
            if a_i.last_hit > timeout:
                to_be_deleted.append(i)
                continue
            else:
                tol_post = a_i.beatInterval*p_tol_post
                tol_pre = a_i.beatInterval*p_tol_pre

                while a_i.prediction + tol_post < e_j:
                    a_i.history.append(a_i.prediction)
                    a_i.prediction += a_i.beatInterval

                if a_i.prediction+tol_pre <= e_j and e_j <= a_i.prediction+tol_post:
                    error = np.abs(e_j - a_i.prediction)
                    err_rel = error/a_i.beatInterval
                    
                    if abs(a_i.prediction - e_j) > tol_inner: 
                        # if prediction is within outer but out of inner tolerance 
                        # --> create new agent based on prediction value (not actual event)
                        new_a_i = a_i.copy()
                        new_agents.append(new_a_i)
                        # without updating beat_interval
                        new_a_i.prediction = a_i.prediction + new_a_i.beatInterval
                        new_a_i.history.append(a_i.prediction)
                        new_a_i.score = (1-err_rel/2)*salience_gaussian(e_j, new_a_i.history[-1])
                        new_a_i.last_hit += 1
                        
                    a_i.beatInterval += error/correction_factor #with updating beat interval
                    a_i.prediction = e_j + a_i.beatInterval
                    a_i.history.append(e_j)
                    a_i.score = (1-err_rel/2)*salience_gaussian(e_j, a_i.history[-1])
        agents+=new_agents
        if len(to_be_deleted)>0:
            print(f"Deleted {len(to_be_deleted)} timeouts in {len(agents)} agents.")
        agents = [agent for i, agent in enumerate(agents) if i not in to_be_deleted]
        agents = remove_duplicate_agents(agents)

    best_agent = choose_best_agent(agents)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))

    # Plot all onsets as vertical lines
    for onset in onsets:
        plt.axvline(x=onset, color='gray', linestyle='--', alpha=0.5)

    # Plot best agent's beat predictions
    plt.vlines(best_agent.history, ymin=0, ymax=1, color='red', label='Predicted Beats')

    plt.title("Onsets vs Predicted Beats")
    plt.xlabel("Time (s)")
    plt.yticks([])  # Remove y-axis ticks (not needed)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    print(f"Beats: {best_agent.history}")
    return best_agent.history
    #return onsets[::10]


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()

    # iterate over input directory
    indir = Path(options.indir)
    infiles = list(indir.glob('*.wav'))
    if tqdm is not None:
        infiles = tqdm.tqdm(infiles, desc='File')
    results = {}
    for filename in infiles:
        results[filename.stem] = detect_everything(filename, options)

    #write output file
    with open(options.outfile, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
    

