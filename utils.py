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
    

def salience(d, p, v, type_="additive", c1=300, c2=-4, c3=1, c4=84, pmin=48, pmax=72):
    def select_p(p):
        if p<= pmin:
            return pmin
        elif pmax <= p:
            return pmax
        else:
            return p
    
    if type_ == "additive":
        return c1*d + c2*select_p(p) + c3*v
    elif type == "multiplicative":
        return d*(c4-select_p(p))*np.log(v)
    else:
        return 1

def salience1(x=None):
    return 1

def salience_ioi_based(onset, onsets, window=0.05):
    index = np.where(onsets == onset)[0][0]
    prev = onsets[index - 1] if index > 0 else onset
    next = onsets[index + 1] if index < len(onsets) - 1 else onset
    ioi_prev = onset - prev
    ioi_next = next - onset
    return 1.0 / (1.0 + abs(ioi_prev - ioi_next))


class beat_agent():
    def __init__(self, beat_interval, pred, hist, score):
        self.beatInterval = beat_interval
        self.prediction = pred
        self.history = hist
        self.score = score
        #self.last_update = hist[0]
        #print(f"beat interval is {beat_interval}")
    
    def copy(self):
        return beat_agent(self.beatInterval, self.prediction, self.history, self.score)


class IOICluster():
    """
    IOI clustering algorithm implementation 
    inspired by "Automatic Extraction of Tempo 
    and Beat from Expressive Performances" by
    Simon Dixon, 2001
    (algortihm for IOI clustering on page 12)
    """
    def __init__(self, cluster_width=0.3):
        self.cluster_width = cluster_width
        self.clusters = [] # initialize empty list for saving our clusters

    def compute_iois(self, onsets):
        iois = []
        for i, e_i in enumerate(onsets):
            for j, e_j in zip(range(i+1, len(onsets)), onsets[i+1:]):
                ioi = np.round(np.abs(e_i-e_j), 5)
                iois.append((i, j, ioi))
        return iois
    
    def cluster_iois(self, iois):
        for i, j, ioi in iois:
            best_cluster = None
            best_distance = float("inf")
            for cluster in self.clusters:
                cluster_mean = np.mean(cluster)
                distance = abs(cluster_mean - ioi)
                if distance < self.cluster_width and distance < best_distance:
                    best_cluster = cluster
                    best_distance = distance
            
            if best_cluster is not None: # if k exists then add IOI_ij to C_k
                # fitting cluster found --> add ioi to this cluster
                best_cluster.append(ioi)
            else: # else create new cluster C_m = {IOI_ij}
                # no fitting cluster found --> create new cluster
                self.clusters.append([ioi]) 
        # remove duplicates --> not necessary!!!!!
        #for i, c_i in enumerate(self.clusters):
        #    self.clusters[i] = list(set(c_i))

        # we do not need to return anything --> we set self values
    
    def merge_clusters(self):
        to_be_deleted = []
        for i, c_i in enumerate(self.clusters):
            c_i_mean = np.mean(c_i)
            for j, c_j in zip(range(i+1, len(self.clusters)), self.clusters[i+1:]):
                c_j_mean = np.mean(c_j)
                if abs(c_i_mean-c_j_mean) < self.cluster_width:
                    c_i+= c_j
                    to_be_deleted.append(j)
        self.clusters = [cluster for i, cluster in enumerate(self.clusters) if i not in to_be_deleted]


    def f(self, d): # relationship function 
        if d >= 1 and d <= 4:
            return 6-d
        elif d >= 5 and d<=8:
            return 1
        else:
            return 0
        
    def f2(self, n):
        # inverse harmonic penalty
        return 1.0 / n
    
    def rank_clusters(self):
        scores = [0]*len(self.clusters)
        for i, c_i in enumerate(self.clusters):
            c_i_mean = np.mean(c_i)
            for j, c_j in enumerate(self.clusters):
                if i == j:
                    continue
                c_j_mean = np.mean(c_j)
                for k in range(1, 5): # harmonic multiples 1 to 4
                    if abs(c_i_mean - k*c_j_mean) < self.cluster_width:
                        scores[i] += self.f(k) * len(c_j)
        return scores
    
    def find_tempi(self, scores):
        ranks = np.argsort(scores)[::-1]
        
        tempo_hypotheses = []
        for i, cluster in enumerate(self.clusters):
            c_mean = np.mean(cluster)
            tempo_hypotheses.append(60/c_mean)
        tempo_hypotheses = np.array(tempo_hypotheses)[ranks]
        indices = np.where((tempo_hypotheses>60) & (tempo_hypotheses<200))
        tempi = tempo_hypotheses[indices]
        return tempi

    def run(self, onsets):
        iois = self.compute_iois(onsets)
        self.cluster_iois(iois)
        self.merge_clusters()
        scores = self.rank_clusters() 
        tempi = self.find_tempi(scores)
        return tempi
    

def choose_best_agent(agents):
    if len(agents) < 1:
        print("No agents")
        return
    
    best_agent = agents[0]
    best_score = best_agent.score
    i = 0
    best_ind = 0
    for agent in agents[1:]:
        if agent.score > best_score:
            best_agent = agent
            best_score = best_agent.score
            best_ind=i
        i+=1
    print(f"The best agent is agent {best_ind}.")

    return best_agent

def remove_duplicate_agents(agents):
    if len(agents)<2:
        return agents
    
    to_be_deleted = []
    for i, agent1 in enumerate(agents):
        for j, agent2 in zip(range(i+1, len(agents)), agents[i+1:]):
            if agent1.beatInterval==agent2.beatInterval and agent1.prediction==agent2.prediction and agent2.history[-5:]==agent2.history[-5:] and agent1.score==agent2.score:
                to_be_deleted.append(j)
    agents = [agent for i, agent in enumerate(agents) if i not in to_be_deleted]
    if len(to_be_deleted)>0:
        print(f"Deleted {len(to_be_deleted)} duplicates")
    return agents


if __name__ == "__main__":
    events = [0, 1.1, 2.0, 3.2, 5.0, 6.1]  # Example event onset times
    cluster_width = 0.3
    ioi_clustering = IOICluster(cluster_width)
    scores = ioi_clustering.run(events)

    for i, cluster in enumerate(ioi_clustering.clusters):
        print(f"Cluster {i}: IOIs = {cluster}, Score = {scores[i]}")





