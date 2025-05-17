import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import mir_eval
from collections import defaultdict
import os
import json


def pad(x, length):
    pad_width = ((0, 0), (0, length - x.shape[1]))
    return np.pad(x, pad_width, mode='constant')

def collate_fn(batch):
    """
    pad inputs of batch with 0s to be of same length
    """
    specs, labels = zip(*batch)
    max_len = max(s.shape[1] for s in specs)

    specs = np.stack([pad(s, max_len) for s in specs])
    labels = np.stack([np.pad(l, (0, max_len - len(l))) for l in labels])
    specs = torch.tensor(specs).unsqueeze(1).float()
    labels = torch.tensor(labels).float()
    
    return specs, labels



def vis_onsets(pred, truths,peak=None):
    for p, tru in zip(pred, truths):
        x = np.arange(len(p))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=np.maximum(0,p), mode='lines'))
        for j, t in enumerate(tru):
            if t != 0:
                fig.add_vline(x=j,line=dict(color='rgba(255, 0, 0, 0.5)'))
        if peak!=None:
            for pe in peak:
                fig.add_vline(x=pe,line=dict(color='rgba(0, 255, 0, 0.5)'))
            fig.show()

        input()

def vis_tempo(autocorrelation, autocorrelation_flux, labels):
    if len(labels)==3:
        bpm = labels[1]
    else:
        bpm = labels[0]
    truth = 60*70/bpm -21

    x = np.arange(len(autocorrelation_flux))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=autocorrelation, mode='lines',name="cnn"))
    fig.add_trace(go.Scatter(x=x, y=autocorrelation_flux, mode='lines',name="flux"))
    fig.add_trace(go.Scatter(x=x, y=autocorrelation_flux+autocorrelation, mode='lines',name="both"))
    fig.add_vline(x=truth,line=dict(color='rgba(255, 0, 0, 0.5)'))
    fig.show()
    input()


def eval_o(model_path, dataset,model_class, odf_rate, sr, hoplength, threshold=0.5):
    """
    get f1-score of data
    model_path:             path of model
    dataset:                data used for evaluation
    model_class:            which model to load
    odf_rate:               same as fps
    sr:
    hoplength:
    threshold:              threshold used for peak detection function
    """

    #load model
    model = model_class()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    all_pred = defaultdict()
    all_true = defaultdict()
    with torch.no_grad():
        for i in range(len(dataset)):
                #get predictions
                data, labels = dataset[i]
                data = torch.tensor(data).unsqueeze(0).unsqueeze(1).float()
                pred = model(data).numpy()[0]
                
                #peak detection fucntion
                #dividing by odf_rate gives the timesteps
                strongest_indices = custom_detect_peaks2(pred, threshold=threshold)/ odf_rate

                #get times of labels
                frame_indices = np.where(labels == 1)[0]
                label_times = librosa.frames_to_time(frame_indices, sr=sr, hop_length=hoplength)

                all_pred[i] = strongest_indices
                all_true[i] = label_times
                #vis_onsets([pred], [labels])                       #visualize detection function + true peaks
                #vis_onsets([pred], [labels], [librosa.time_to_frames(strongest_indices, sr=sr, hop_length=hoplength)])  #visualize detection function + true peaks + perdicted peaks
    
    #return f1-score
    return eval_onsets(all_true, all_pred)

def eval_t(model_path, dataset,modelclass):
    """
    compute tempo estimate
    model_path:             path of model
    dataset:                data used for evaluation
    model_class:            which model to load
    """

    #load model
    model = modelclass()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()


    all_pred = defaultdict()
    all_true = defaultdict()

    with torch.no_grad():
        for i in range(len(dataset)):
            #get detection function
            data, labels = dataset[i]
            tensor_data = torch.tensor(data).unsqueeze(0).unsqueeze(1).float()
            pred = model(tensor_data).numpy()[0]

            #compute autocorrelation of detection function
            autocorrelation = np.array([np.dot(pred[:-tau], pred[tau:]) for tau in range(21, 70)])          #autocorrelation
            autocorrelation = autocorrelation/np.max(autocorrelation)                                       #devide by max to get values in range [0,1]
            
            #compute autocorrelation of spectral flux
            h=1
            sd_t = data[:, h:] - data[:, :int(data.shape[1])-h]
            hw_t = np.maximum(0,sd_t)**2
            dSD_t = np.sum(hw_t, axis=0)
            autocorrelation_flux = np.array([np.dot(dSD_t[:-tau], dSD_t[tau:]) for tau in range(21, 70)])   #autocorrelation
            autocorrelation_flux = autocorrelation_flux/np.max(autocorrelation_flux)                        #devide by max to get values in range [0,1]

            #vis_tempo(autocorrelation, autocorrelation_flux, labels)                                       #visualize the two autocorrelation functions and the true tempo

            #IOI histogram approach: TODO: not fully tested
            """onset_times = custom_detect_peaks2(pred, threshold=0.4)
            onset_times = np.array(onset_times)

            diff_matrix = onset_times[None, :] - onset_times[:, None]
            iois = diff_matrix[np.triu_indices(len(onset_times), k=1)]
            iois = iois[(iois >= 21) & (iois <= 70)]
            #print(iois)

            ioi = np.array([np.sum(iois==freq) for freq in range(21, 70)])
            ioi =  ioi/np.max(ioi)"""

            """x = np.arange(len(ioi))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=ioi, mode='lines'))
            fig.show()
            print(labels)
            input()"""

            #get index of autocorerlation
            #NOTE: converting the index to bpm one has to add 21 to the index since the autocorrelation function starts with 0
            #      and devide 60(seconds)*70(fps)/index to get the tempo
            tempo = 4200/(np.argmax(autocorrelation)+21)
            #add a second estimate of half the tempo
            all_pred[i] = [tempo/2, tempo]
            all_true[i] = labels
           

    return eval_tempo(all_true, all_pred)

def eval_b():
    """
    compute beats estimates
    """
    all_pred = defaultdict()
    all_true = defaultdict()
    #TODO
    
    return eval_beats(all_true, all_pred)


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

    adapt_pred = np.maximum(adapt_pred,0)
    strongest_indices = []
    for i, (prev, curr, nex) in enumerate(zip(adapt_pred[:len(adapt_pred-2)],adapt_pred[1:len(adapt_pred-1)],adapt_pred[2:])):
        adapt_threshold = threshold + lamb * np.median(pred[max(0, i - win_size):min(len(pred), i + win_size + 1)])
        if curr > prev and curr > nex and curr >= adapt_threshold:
            strongest_indices.append(i+1)
    
    return np.array(strongest_indices)


def eval_onsets(truth, preds):
    #from given script
    """
    Computes the average onset detection F-score.
    """
    return sum(mir_eval.onset.f_measure(np.asarray(truth[k]),
                                        np.asarray(preds[k]),
                                        0.05)[0]
               for k in truth if k in preds) / len(truth)

def eval_tempo(truth, preds):
    #from given script
    """
    Computes the average tempo estimation p-score.
    """
    def prepare_truth(tempi):
        if len(tempi) == 3:
            tempi, weight = tempi[:2], tempi[2]
        else:
            tempi, weight = [tempi[0] / 2., tempi[0]], 0.
        return np.asarray(tempi), weight

    def prepare_preds(tempi):
        if len(tempi) < 2:
            tempi = [tempi[0] / 2., tempi[0]]
        return np.asarray(tempi)

    return sum(mir_eval.tempo.detection(*prepare_truth(truth[k]),
                                        prepare_preds(preds[k]),
                                        0.08)[0]
               for k in truth if k in preds) / len(truth)

def eval_beats(truth, preds):
    """
    Computes the average beat detection F-score.
    """
    return sum(mir_eval.beat.f_measure(np.asarray(truth[k]['beats']),
                                       np.asarray(preds[k]['beats']),
                                       0.07)
               for k in truth if k in preds) / len(truth)


def predict_o(model_path, dataset,model_class, odf_rate, outfile):
    """
    get the predictions for the challenge server
    """
    #load model
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

                all_pred[str(os.path.splitext(os.path.basename(filename))[0])] = {'onsets': list(np.round(strongest_indices, 3))}
    with open(outfile, 'w') as f:
        json.dump(all_pred, f)

def predict_t(model_path, dataset,model_class, outfile):
    """
    get the predictions for the challenge server
    """
    #load model
    model = model_class()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    all_pred = defaultdict()

    with torch.no_grad():
        for i in range(len(dataset)):
                data, filename = dataset[i]
                data = torch.tensor(data).unsqueeze(0).unsqueeze(1).float()
                pred = model(data).numpy()[0]

                autocorrelation = [np.dot(pred[:-tau], pred[tau:]) for tau in range(21, 70)]
                tempo = 4200/(np.argmax(autocorrelation)+21)

                all_pred[str(os.path.splitext(os.path.basename(filename))[0])] = {'tempo': [tempo/2, tempo]}
    with open(outfile, 'w') as f:
        json.dump(all_pred, f)


def predict_b():
    """
    get predictions for beat
    """

    #TODO
    pass
