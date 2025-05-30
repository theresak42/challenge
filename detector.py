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

from data import *
from model import *
from utils import *
from train import train

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss



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
    """
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

    return dSD_t, values_per_second
    """
    return


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


    """
    norm_odf = (odf - np.min(odf)) / (np.max(odf) - np.min(odf))
    #print(type(norm_odf), norm_odf.shape)
    #print(type(norm_odf_smooth), norm_odf_smooth.shape)
    #TODO: use smoothing function/ DC removal
    t = "F"
    if t=="T":
        plt.figure(figsize=(10, 4))
        plt.plot(norm_odf, color='dodgerblue', linewidth=2)
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
    for i, (prev, curr, nex) in enumerate(zip(norm_odf[:len(norm_odf-2)],norm_odf[1:len(norm_odf-1)],norm_odf[2:])):
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
    """
    data_split = 0.8
    fps =70

    data, labels, sr, hoplength = load_data(indir,methode="onsets", use_extra=True, fps=fps)
    
    print("Data loaded")

    len_data = int(len(data)*data_split)
    complete_dataset = AudioDataset(data, labels)
    train_dataset = AudioDataset(data[:len_data], labels[:len_data])
    val_dataset = AudioDataset(data[len_data:], labels[len_data:])

    

    train_dataloader = DataLoader(train_dataset, 4, shuffle= True, collate_fn = collate_fn)
    val_dataloader = DataLoader(val_dataset, 4, shuffle= False, collate_fn = collate_fn)
    print("Dataloader ready")

    tempo_data, tempos, *_ = load_data(indir, methode="tempo", use_extra=False, fps = fps)
    tempo_dataset = AudioDataset(tempo_data, tempos)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    model = CNN_model2().to(device)

    num_epochs = 35

    optimizer = AdamW(model.parameters(), lr = 0.0001)

    loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([24.0]).to(device))


    #train(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs, device=device, save_model=True, model_name="cnn3")

    model_path = r"\trained_models\cnn2.pt"

    #evaluate onsets
    #f1 = eval_o(model_path,complete_dataset,CNN_model2,fps,sr,hoplength, threshold=0.4)
    #print(f1)

    #evaluate tempos
    p_score = eval_t(model_path, tempo_dataset, CNN_model2)
    print(p_score)



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

    #write output file
    with open(options.outfile, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
    

