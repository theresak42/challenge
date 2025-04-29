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
import scipy
from scipy.io import wavfile

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

        #from utils import vis
        #vis([onsets], [signal])

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

    values, values_per_second = LFSF(sample_rate, signal, fps, spect, magspect, melspect)
    """h = 1
    x_t = np.log10(1+10*magspect)
    print(x_t.shape)
    sd_t = x_t[:, h:] - x_t[:, :int(x_t.shape[1])-h]
    print(sd_t.shape)
    hw_t = np.maximum(0,sd_t)**2
    print(hw_t.shape)
    dSD_t = np.sum(hw_t, axis=0)

    values = dSD_t
    values_per_second = fps
"""
    #print(dSD_t.shape)
    

    """plt.figure(figsize=(10, 4))
    plt.plot(dSD_t, color='dodgerblue', linewidth=2)
    plt.title("Onset Strength Over Time")
    plt.xlabel("Time Frame (t)")
    plt.ylabel("Strength")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pic.png")
    plt.close()"""

    #input()

    return values, values_per_second


def LFSF(sample_rate, signal, fps, spect, magspect, melspect, window_size=23, hop_size=10, lam=1, n_fft=2048):
    """
    Onset detection using the LogFiltSpectFlux algorithm.
    """
    
    """
    X_k = librosa.stft(
            signal, hop_length=hop_size, win_length=window_size, window='hann', n_fft=n_fft)
    
    magnitudes=[]
    semitone_filterbank, sample_rates = librosa.filters.semitone_filterbank()
    for cur_sr, cur_filter in zip(sample_rates, semitone_filterbank):
        w, h = scipy.signal.sosfreqz(cur_filter,fs=cur_sr, worN=n_fft/2+1)
        print(w,h)
        magnitudes.append(20 * np.log10(1+lam*np.abs(h)))
    magnitudes = np.array(magnitudes)
    print(magnitudes.shape)
    magspect = np.dot(np.abs(X_k), magnitudes)
    """

    h = 1
    X_t = np.log10(1+10*magspect)
    #print(f"X_t: {X_t[0]}")
    distances = X_t[:, h:] - X_t[:, :int(X_t.shape[1])-h]
    #print(f"distances: {distances[0]}")
    values = np.sum(np.maximum(distances, 0), axis=0)
    print(values[:10])
    #print(f"\nreturn values shapes: {values.shape}, fps={fps}")

    return values, fps


def detect_onsets(odf_rate, odf, options):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """
    # we only have a dumb dummy implementation here.
    # it returns the timestamps of the 100 strongest values.
    # this is not a useful solution at all, just a placeholder.
    strongest_indices = np.argpartition(odf, 100)[:100]
    strongest_indices.sort()

    norm_odf = (odf - np.min(odf)) / (np.max(odf) - np.min(odf))
    #TODO: use smoothing function/ DC removal

    #implement custom peak picking

    threshold = 0.15
    strongest_indices = []
    for i, (prev, curr, nex) in enumerate(zip(norm_odf[:len(norm_odf-2)],norm_odf[1:len(norm_odf-1)],norm_odf[2:])):
        if curr > prev and curr > nex and curr > threshold:
            strongest_indices.append(i+1)

    strongest_indices = np.array(strongest_indices)
    print(strongest_indices/odf_rate, len(strongest_indices))
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
    tempo = 60 / (onsets[1] - onsets[0])
    return [tempo / 2, tempo]


def detect_beats(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, tempo, options):
    """
    Detect beats using any of the input representations.
    Returns the positions of all beats in seconds.
    """
    # we only have a dumb dummy implementation here.
    # it returns every 10th onset as a beat.
    # this is not a useful solution at all, just a placeholder.
    return onsets[::10]


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

    # write output file
    #with open(options.outfile, 'w') as f:
        #json.dump(results, f)


if __name__ == "__main__":
    main()

