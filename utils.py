import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import mir_eval
from collections import defaultdict

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
    print(np.unique(labels))
    input()
    specs = torch.tensor(specs).unsqueeze(1).float()
    labels = torch.tensor(labels).float()
    
    return specs, labels


def pred_to_time(pred, fps):
      test = librosa.frames_to_time


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

def eval(model_path, dataset, odf_rate, sr, hoplength, device):
     model = CNN_model()
     model.load_state_dict(torch.load(model_path, weights_only=True))
     model.eval()
     all_pred = defaultdict()
     all_true = defaultdict()
     with torch.no_grad():
        for i in range(len(dataset)):
                data, labels = dataset[i]
                data = torch.tensor(data).unsqueeze(0).unsqueeze(1).float()
                pred = model(data).numpy()[0]

                #pred = smoothing(pred, window=5)

                strongest_indices = find_peaks_cwt(pred, widths=1)/ odf_rate

                frame_indices = np.where(labels == 1)[0]
                label_times = librosa.frames_to_time(frame_indices, sr=sr, hop_length=hoplength)
                #print(strongest_indices)
                #print(label_times)
                all_pred[i] = strongest_indices
                all_true[i] = label_times
                #vis([pred], [labels])
                #input()

     return eval_onsets(all_true, all_pred)






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