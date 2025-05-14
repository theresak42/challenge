import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import mir_eval
from collections import defaultdict
import os
import json

from scipy.signal import find_peaks_cwt
from model import *

def pad(x, length):
        pad_width = ((0, 0), (0, length - x.shape[1]))
        return np.pad(x, pad_width, mode='constant')

def collate_fn(batch):
    specs, labels = zip(*batch)
    max_len = max(s.shape[1] for s in specs)

    specs = np.stack([pad(s, max_len) for s in specs])
    labels = np.stack([np.pad(l, (0, max_len - len(l))) for l in labels])
    specs = torch.tensor(specs).unsqueeze(1).float()
    labels = torch.tensor(labels).float()
    
    return specs, labels



def vis(pred, truths):
    for p, tru in zip(pred, truths):
        x = np.arange(len(p))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=np.maximum(0,p), mode='lines'))
        for j, t in enumerate(tru):
            if t != 0:
                fig.add_vline(x=j,line=dict(color='rgba(255, 0, 0, 0.5)'))
        fig.show()

        input()

def vis_2(pred, truths, peaks):
    for p, tru, peak in zip(pred, truths, peaks):
        x = np.arange(len(p))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=np.maximum(0,p), mode='lines'))
        for j, t in enumerate(tru):
            if t != 0:
                fig.add_vline(x=j,line=dict(color='rgba(255, 0, 0, 0.5)'))
        for pe in peak:
             fig.add_vline(x=pe,line=dict(color='rgba(0, 255, 0, 0.5)'))
        fig.show()

        input()

def eval(model_path, dataset,modelclass, odf_rate, sr, hoplength, threshold=0.5):


    model = modelclass()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    all_pred = defaultdict()
    all_true = defaultdict()
    with torch.no_grad():
        for i in range(len(dataset)):
                data, labels = dataset[i]
                data = torch.tensor(data).unsqueeze(0).unsqueeze(1).float()
                pred = model(data).numpy()[0]

                strongest_indices = custom_detect_peaks2(pred, threshold=threshold)/ odf_rate

                frame_indices = np.where(labels == 1)[0]
                label_times = librosa.frames_to_time(frame_indices, sr=sr, hop_length=hoplength)
                #print(strongest_indices)
                #print(label_times)
                all_pred[i] = strongest_indices
                all_true[i] = label_times
                #vis([pred], [labels])
                #vis_2([pred], [labels], [librosa.time_to_frames(strongest_indices, sr=sr, hop_length=hoplength)])
                #input()

    return eval_onsets(all_true, all_pred)


def custom_detect_peaks(pred, threshold = 0.5):
    #fixed threshold
    strongest_indices = []
    for i, (prev, curr, nex) in enumerate(zip(pred[:len(pred-2)],pred[1:len(pred-1)],pred[2:])):
        if curr > prev and curr > nex and curr > threshold:
            strongest_indices.append(i+1)
    
    return np.array(strongest_indices)

def custom_detect_peaks2(pred, threshold = 0.5, windowsize = 2):
    #fixed threshold + smoothing
    half_windowsize = windowsize // 2

    smooth_pred = np.array([
        np.mean(pred[max(0, i - half_windowsize):min(len(pred), i + half_windowsize + 1)])
        for i in range(len(pred))
    ])

    # 0.5, 2:   0.8639739355842295
    # 0.4  2:   0.8660277668309816
    # 0.3, 2:   0.865986292083449


    strongest_indices = []
    for i, (prev, curr, nex) in enumerate(zip(smooth_pred[:len(smooth_pred-2)],smooth_pred[1:len(smooth_pred-1)],smooth_pred[2:])):
        if curr > prev and curr > nex and curr > threshold:
            strongest_indices.append(i+1)
    
    return np.array(strongest_indices)


def custom_detect_peaks3(pred, threshold = 0.5, windowsize = 2):
    #adaptive threshold + smoothing
    half_windowsize = windowsize // 2

    adapt_pred = np.array([
        np.mean(pred[max(0, i - half_windowsize):min(len(pred), i + half_windowsize + 1)])
        for i in range(len(pred))
    ])

    lamb = 0.2
    win_size = 2
    # lamb = 0.4, winsize 1: 0.8620492937521748
    # lamb = 0.4, winsize 2: 0.8682685388439372
    # lamb = 0.4, winsize 3: 0.8657293278118826
    # lamb = 0.4, winsize 4: 0.8653448114013235

    # lamb = 0.3, winsize 2: 0.8687202666562578
    # lamb = 0.25, winsize 2: 0.8695188520185966
    # lamb = 0.2, winsize 2: 0.8699503678113406                       best model
    # lamb = 0.15, winsize 2: 0.8689458075320877
    # lamb = 0.1, winsize 2: 0.867848788083857

    adapt_pred = np.maximum(adapt_pred,0)
    strongest_indices = []
    for i, (prev, curr, nex) in enumerate(zip(adapt_pred[:len(adapt_pred-2)],adapt_pred[1:len(adapt_pred-1)],adapt_pred[2:])):
        adapt_threshold = threshold + lamb * np.median(pred[max(0, i - win_size):min(len(pred), i + win_size + 1)])
        if curr > prev and curr > nex and curr >= adapt_threshold:
            strongest_indices.append(i+1)
    
    return np.array(strongest_indices)


def eval_onsets(truth, preds):
    """
    Computes the average onset detection F-score.
    """
    return sum(mir_eval.onset.f_measure(np.asarray(truth[k]),
                                        np.asarray(preds[k]),
                                        0.05)[0]
               for k in truth if k in preds) / len(truth)


def smoothing(values, window=3):

    avg = np.convolve(values, np.ones(window)/window, mode='same')
    return avg


def predict(model_path, dataset,model_class, odf_rate, outfile):
    model = model_class()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    all_pred = defaultdict()

    with torch.no_grad():
        for i in range(len(dataset)):
                data, filename = dataset[i]
                data = torch.tensor(data).unsqueeze(0).unsqueeze(1).float()
                pred = model(data).numpy()[0]

                strongest_indices = custom_detect_peaks2(pred, threshold=0.4)/ odf_rate

                #print(filename, strongest_indices)

                all_pred[str(os.path.splitext(os.path.basename(filename))[0])] = {'onsets': list(np.round(strongest_indices, 3))}
    with open(outfile, 'w') as f:
        json.dump(all_pred, f)
    
