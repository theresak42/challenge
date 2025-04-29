import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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


def pred_to_time(pred, fps):
      test = librosa.frames_to_time


def vis(pred, truths):
    for p, tru in zip(pred, truths):
        #print(sum(tru))
        x = np.arange(len(p))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=np.maximum(0,p), mode='lines'))
        for j, t in enumerate(tru):
            if t != 0:
                fig.add_vline(x=j,line=dict(color='rgba(255, 0, 0, 0.5)'))
        fig.show()

        input()
    
