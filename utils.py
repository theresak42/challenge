import torch
import numpy as np

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