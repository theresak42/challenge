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

def vis_beats(true_onsets, pred_onsets, true_beats, pred_beats, title="Onsets and Beats"):
    """
    Plots true and predicted onsets and beats using Plotly.

    Parameters:
    - true_onsets: list of float, time in seconds
    - pred_onsets: list of float, time in seconds
    - true_beats: list of float, time in seconds
    - pred_beats: list of float, time in seconds
    - title: string, plot title
    """
    
    fig = go.Figure()

    # True Onsets
    if true_onsets is not None:
        for t in true_onsets:
            fig.add_trace(go.Scatter(
                x=[t, t],
                y=[0, 1],
                mode="lines",
                name="True Onset",
                line=dict(color="green", dash="solid"),
                showlegend=False  # Avoid multiple entries in legend
            ))
        #if true_onsets.any():
        #    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="True Onset", line=dict(color="green", dash="solid")))

    # Predicted Onsets
    if pred_onsets is not None:
        for t in pred_onsets:
            fig.add_trace(go.Scatter(
                x=[t, t],
                y=[0, 1],
                mode="lines",
                name="Predicted Onset",
                line=dict(color="red", dash="dash"),
                showlegend=False
            ))
    #if pred_onsets:
    #    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Predicted Onset", line=dict(color="red", dash="dash")))

    # True Beats
    if true_beats is not None:
        for t in true_beats:
            fig.add_trace(go.Scatter(
                x=[t, t],
                y=[0, 1],
                mode="lines",
                name="True Beat",
                line=dict(color="blue", dash="solid"),
                showlegend=False
            ))
    #if true_beats:
    #    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="True Beat", line=dict(color="blue", dash="solid")))

    # Predicted Beats
    if pred_beats is not None:
        for t in pred_beats:
            fig.add_trace(go.Scatter(
                x=[t, t],
                y=[0, 1],
                mode="lines",
                name="Predicted Beat",
                line=dict(color="orange", dash="dash"),
                showlegend=False
            ))
    #if pred_beats:
    #    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Predicted Beat", line=dict(color="orange", dash="dash")))

    # Layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300
    )

    fig.show()


def eval_o(model_path, dataset,model_class, odf_rate, sr, hoplength, threshold=0.5, type_="onsets"):
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

                #if type_ == "beats":
                    #print(strongest_indices)
                    #strongest_indices = multi_agent_post(strongest_indices)
                    #pass
                    

                #get times of labels
                frame_indices = np.where(labels == 1)[0]
                label_times = librosa.frames_to_time(frame_indices, sr=sr, hop_length=hoplength)

                all_pred[i] = strongest_indices
                all_true[i] = label_times
                #if len(label_times) < 2:
                #    print(f"Sample {i}, truths: {label_times}")
                #print(f"Sample {i} predicts {len(all_pred[i])} beats: {all_pred[i][:10]}")
                #vis_onsets([strongest_indices], [labels])                       #visualize detection function + true peaks
                #vis_onsets([pred], [labels], [librosa.time_to_frames(strongest_indices, sr=sr, hop_length=hoplength)])  #visualize detection function + true peaks + perdicted peaks
                #vis_beats(None, None, label_times, strongest_indices)
                #input()

    #return f1-score
    if type_ == "beats":
        return eval_beats(all_true, all_pred)
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
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
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

def eval_b(dataset, onsets_all, tempo_all, beats_all=None):
    """
    compute beats estimates
    """
    all_pred = defaultdict()
    all_true = defaultdict()
    
    for i in range(len(dataset)):
        #get predictions
        data, labels = dataset[i]
        onsets = onsets_all[i]
        tempo = tempo_all[i]

        beats=None
        if beats_all is not None:
            beats = beats_all[i]
        
        all_pred[i] = predict_b3(onsets, tempo, beats)
        all_true[i] = labels

        #vis_beats(true_onsets=None, pred_onsets=onsets, true_beats=labels, pred_beats=all_pred[i])
        #input()

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
    #print(f"Beat evaluation")
    #print(f"truth: {truth}")
    #print(f"predictions: {preds}")
    
    return sum(mir_eval.beat.f_measure(np.asarray(truth[k]),
                                       np.asarray(preds[k]),
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

def predict_b2(model_path, dataset,model_class, odf_rate, outfile, threshold=0.5):
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

                strongest_indices = custom_detect_peaks2(pred, threshold=threshold)/ odf_rate

                all_pred[str(os.path.splitext(os.path.basename(filename))[0])] = {'beats': list(np.round(strongest_indices, 3))}
    with open(outfile, 'w') as f:
        json.dump(all_pred, f)


def predict_b(onsets, tempo):
    """
    get predictions for beat
    """
    startup_period = len(onsets)//10
    timeout = 3
    max_misses = 20#len(onsets)//20 +2
    max_number_agents = 1000
    correction_factor = 5
    tol_inner = 40./1000. # s = 40 ms
    p_tol_pre = 0.1
    p_tol_post= 0.2

    # Initialization
    agents = []
    for t_i in tempo:
        if t_i > 0:
            for e_j in onsets[:startup_period]:
                agent = beat_agent(beat_interval=60/t_i, 
                                pred = e_j+60/t_i, 
                                hist = [e_j], 
                                score = float("inf"),
                                ibi=[])
                agents.append(agent)
    #print(f"Initialized {len(agents)} agents.")
    if len(agents) ==0:
        print(f"Tempi: {tempo}")
        print(f"Onset len: {len(onsets)}")

    # Main loop
    for _, e_j in enumerate(onsets):
        new_agents = []
        to_be_deleted = []    
        for i, a_i in enumerate(agents):
            if e_j - a_i.history[-1] > timeout:
                to_be_deleted.append(i)
                continue
            if a_i.misses > max_misses:
                to_be_deleted.append(i)
                continue
            else:
                tol_post = a_i.beatInterval*p_tol_post
                tol_pre = a_i.beatInterval*p_tol_pre

                while a_i.prediction + tol_post < e_j:
                    if not(a_i.prediction+a_i.beatInterval+tol_pre <= e_j and e_j <= a_i.prediction+a_i.beatInterval+tol_post):
                        a_i.history.append(a_i.prediction)
                    a_i.prediction += a_i.beatInterval
                    a_i.misses +=1

                if a_i.prediction+tol_pre <= e_j and e_j <= a_i.prediction+tol_post:
                    error = e_j - a_i.prediction
                    #err_rel = error/a_i.beatInterval
                    
                    if abs(a_i.prediction - e_j) > tol_inner: 
                        # if prediction is within outer but out of inner tolerance 
                        # --> create new agent based on prediction value (not actual event)
                        new_a_i = a_i.copy()
                        # without updating beat_interval
                        new_a_i.prediction = a_i.prediction + new_a_i.beatInterval
                        new_a_i.history.append(a_i.prediction)
                        #new_a_i.score = (1-err_rel/2)*salience_gaussian(e_j, new_a_i.history[-1])
                        new_a_i.ibi.append(a_i.beatInterval)
                        new_a_i.score=salience_variance(new_a_i.ibi)
                        #new_a_i.score=score_agent(new_a_i)
                        new_a_i.misses += 1
                        new_agents.append(new_a_i)
                        
                    
                    #if error/correction_factor < a_i.beatInterval:
                    a_i.beatInterval -= error/correction_factor #with updating beat interval
                    #if a_i.beatInterval <= 0:
                    #    a_i.beatInterval = 1
                    a_i.prediction = e_j + a_i.beatInterval
                    a_i.history.append(e_j)
                    a_i.ibi.append(a_i.beatInterval)
                    #a_i.score = (1-err_rel/2)*salience_gaussian(e_j, a_i.history[-1])
                    a_i.score = salience_variance(a_i.ibi)
                    #a_i.score = score_agent(a_i)
        #print(f"Number agents: {len(agents)}")
        #if len(to_be_deleted)>0:
        #    print(f"to be del: {len(to_be_deleted)}")
        if len(to_be_deleted)>0 and len(agents) > len(to_be_deleted):
            #print(f"Deleted {len(to_be_deleted)} timeouts in {len(agents)} agents.")
            agents = [agent for i, agent in enumerate(agents) if i not in to_be_deleted]
        agents+=new_agents
        agents = remove_duplicate_agents(agents)
        if len(agents)>max_number_agents: #too many agents, do a reset
            print(f"Agent reset")
            agents = [choose_best_agent(agents)]
        
    print(f"Choose best agent from {len(agents)} candidates")
    best_agent = choose_best_agent(agents)

    #print(f"Beats: {best_agent.history}")

    #for i in range(1, len(best_agent.history)):
    #    if best_agent.history[i] <= best_agent.history[i-1]:
    #        print(best_agent.history)
    #        raise(NotImplementedError())
    predicted_beats = best_agent.history
    #predicted_beats = np.round(np.array(predicted_beats), decimals=5)
    #predicted_beats = np.sort(np.unique(predicted_beats))
    return predicted_beats
    

def predict_b3(onsets, tempo, beat_candidates=None):
    """
    get predictions for beat
    """
    startup_period = len(onsets)//10
    timeout = 3
    max_misses = 20#len(onsets)//20 +2
    max_number_agents = 1000
    correction_factor = 2
    tol_inner = 40./1000. # s = 40 ms
    p_tol_pre = 0.2
    p_tol_post= 0.2

    if beat_candidates is None:
        print("Use onsets as candidates")
        beat_candidates = onsets

    # Initialization
    agents = []
    for t_i in tempo:
        for e_j in onsets[:startup_period]:
            agent = beat_agent(beat_interval=60.0/t_i, 
                            pred = e_j+60.0/t_i, 
                            hist = [e_j], 
                            score = float("-inf"),
                            ibi=[])
            agents.append(agent)
    #print(f"Initialized {len(agents)} agents.")

    # assert there are agents
    if len(agents) ==0:
        print(f"Tempi: {tempo}")
        print(f"Onset len: {len(onsets)}")

    # Main loop
    for _, e_j in enumerate(onsets):
        new_agents = []
        #to_be_deleted = []    
        for i, a_i in enumerate(agents):
            #if e_j - a_i.history[-1] > timeout:
                #to_be_deleted.append(i)
                #continue
            #    pass
            #if a_i.misses > max_misses:
                #to_be_deleted.append(i)
                #continue
            #    pass
            #else:
            tol_post = a_i.beatInterval*p_tol_post
            tol_pre = a_i.beatInterval*p_tol_pre

            while a_i.prediction + tol_post < e_j:
                if not(a_i.prediction+a_i.beatInterval+tol_pre <= e_j and e_j <= a_i.prediction+a_i.beatInterval+tol_post):
                    a_i.history.append(a_i.prediction)
                a_i.prediction += a_i.beatInterval
                a_i.misses +=1
                a_i.score = overlap_score(beat_candidates, a_i.history)

            if a_i.prediction+tol_pre <= e_j and e_j <= a_i.prediction+tol_post:
                error = e_j - a_i.prediction
                #err_rel = error/a_i.beatInterval
                
                if abs(a_i.prediction - e_j) > tol_inner: 
                    # if prediction is within outer but out of inner tolerance 
                    # --> create new agent based on prediction value (not actual event)
                    new_a_i = a_i.copy()
                    # without updating beat_interval
                    new_a_i.prediction = a_i.prediction + new_a_i.beatInterval
                    new_a_i.history.append(a_i.prediction)
                    new_a_i.ibi.append(a_i.beatInterval)
                    #new_a_i.score=salience_variance(new_a_i.ibi)
                    new_a_i.score = overlap_score(beat_candidates, new_a_i.history)
                    new_a_i.misses += 1
                    new_agents.append(new_a_i)
                 
                if error > 0:
                    print(f"adjusted {a_i.beatInterval} by {error/correction_factor}")
                a_i.beatInterval -= error/correction_factor #with updating beat interval
                a_i.prediction = e_j + a_i.beatInterval
                a_i.history.append(e_j)
                a_i.ibi.append(a_i.beatInterval)
                #a_i.score = salience_variance(a_i.ibi)
                a_i.score = overlap_score(beat_candidates, a_i.history)
        #print(f"Number agents: {len(agents)}")
        #if len(to_be_deleted)>0:
        #    print(f"to be del: {len(to_be_deleted)}")
        #if len(to_be_deleted)>0 and len(agents) > len(to_be_deleted):
            #print(f"Deleted {len(to_be_deleted)} timeouts in {len(agents)} agents.")
        #    agents = [agent for i, agent in enumerate(agents) if i not in to_be_deleted]
        #agents+=new_agents
        agents = remove_duplicate_agents(agents)
        if len(agents)>max_number_agents: #too many agents, do a reset
            print(f"Agent reset")
            agents = [choose_best_agent(agents)]
        
    #print(f"Choose best agent from {len(agents)} candidates")
    best_agent = choose_best_agent(agents, min_=False)

    #print(f"Beats: {best_agent.history}")

    #for i in range(1, len(best_agent.history)):
    #    if best_agent.history[i] <= best_agent.history[i-1]:
    #        print(best_agent.history)
    #        raise(NotImplementedError())
    predicted_beats = best_agent.history
    #predicted_beats = np.round(np.array(predicted_beats), decimals=5)
    #predicted_beats = np.sort(np.unique(predicted_beats))
    return predicted_beats

def get_onsets(model_path, dataset, model_class, odf_rate, sr, hoplength, threshold=0.5, device=torch.device('cpu')):
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
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    all_pred = defaultdict()
    with torch.no_grad():
        for i in range(len(dataset)):
                #get predictions
                data, labels = dataset[i]
                data = torch.tensor(data).unsqueeze(0).unsqueeze(1).float()
                pred = model(data).numpy()[0]
                
                #peak detection fucntion
                #dividing by odf_rate gives the timesteps
                strongest_indices = custom_detect_peaks2(pred, threshold=threshold)/ odf_rate
                
                all_pred[i] = strongest_indices
    return all_pred

def get_tempo(model_path, dataset,modelclass):
    """
    compute tempo estimate
    model_path:             path of model
    dataset:                data used for evaluation
    model_class:            which model to load
    """

    #load model
    model = modelclass()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()


    all_pred = defaultdict()

    with torch.no_grad():
        for i in range(len(dataset)):
            #get detection function
            data, labels = dataset[i]
            
            # test multiple-agent approach with perfect tempo hypothesis
            #if len(labels)==3:
            #    all_pred[i] = [labels[0] if labels[2]>=0.5 else labels[1]]
            #else:
            #    all_pred[i] = [labels[0]]
            #continue
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

            #get index of autocorerlation
            #NOTE: converting the index to bpm one has to add 21 to the index since the autocorrelation function starts with 0
            #      and devide 60(seconds)*70(fps)/index to get the tempo
            tempo = 4200/(np.argmax(autocorrelation)+21)
            #add a second estimate of half the tempo
            all_pred[i] = [tempo/2, tempo]
            
    return all_pred


"""
Beat detection helper functions
"""

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
    assert(isinstance(onsets, np.ndarray))
    index = np.where(onsets == onset)[0][0]
    prev = onsets[index - 1] if index > 0 else onset
    next = onsets[index + 1] if index < len(onsets) - 1 else onset
    ioi_prev = onset - prev
    ioi_next = next - onset
    return 1.0 / (1.0 + abs(ioi_prev - ioi_next))

def salience_interval_based(interval, window=0.05):
    return 1.0 / (1.0 + abs(interval))

def salience_gaussian(onset, expected_time, std=0.05):
    return np.exp(-((onset - expected_time) ** 2) / (2 * std ** 2))

def salience_variance(ibi):
    # an agent is the better, the more invariant the beats are
    return 1-np.var(ibi)

def score_agent(agent):
    # TODO
    score=None
    return score


class beat_agent():
    def __init__(self, beat_interval, pred, hist, score, ibi, misses=0):
        self.beatInterval = beat_interval
        self.prediction = pred
        self.history = hist
        self.score = score
        self.misses = misses
        self.ibi = ibi
        #print(f"beat interval is {beat_interval}")
    
    def copy(self):
        return beat_agent(self.beatInterval, self.prediction.copy(), self.history.copy(), self.score, self.ibi.copy(), self.misses)


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
    

def choose_best_agent(agents, min_=True):
    if len(agents) < 1:
        #print("No agents")
        return
    best_agent = agents[0]
    best_score = best_agent.score
    i = 0
    best_ind = 0
    
    if min_:
        for agent in agents[1:]:
            if agent.score < best_score:
                best_agent = agent
                best_score = best_agent.score
                #best_ind=i
            i+=1
    else:
        for agent in agents[1:]:
            #print(f"Agent {i} has score {agent.score}")
            if agent.score > best_score:
                best_agent = agent
                best_score = best_agent.score
                best_ind=i
            i+=1
    print(f"The best agent is agent {best_ind} with score {best_score}.")

    return best_agent

def remove_duplicate_agents(agents, view=1, eps=0.00001):
    if len(agents)<2:
        return agents
    
    to_be_deleted = []
    for i, agent1 in enumerate(agents):
        for j, agent2 in zip(range(i+1, len(agents)), agents[i+1:]):
            if i in to_be_deleted or j in to_be_deleted:
                continue
            if abs(agent1.beatInterval-agent2.beatInterval)<eps and (agent1.prediction-agent2.prediction)<eps and (agent1.history[-1]-agent2.history[-1])<eps:# and agent1.score==agent2.score:
                """
                if len(agent1.history) >= len(agent2.history):
                    to_be_deleted.append(j) # keep the agent with the longer history
                else:
                    to_be_deleted.append(i)
                """
                if agent1.score >= agent2.score:
                    to_be_deleted.append(j) # keep the agent with the higher score
                else:
                    to_be_deleted.append(i)
                
                #to_be_deleted.append(j)
    if len(to_be_deleted) < len(agents):
        agents = [agent for i, agent in enumerate(agents) if i not in to_be_deleted]
    else: 
        agents = [agents[0]]
    #    agents = [choose_best_agent(agents)]
    #if len(agents)==0:
    #    print("No agents left")
    if len(to_be_deleted)>0:
        print(f"Deleted {len(to_be_deleted)} duplicates")
    return agents

def beats_postprocessing(indices, eps=0.001):
    ibis = np.diff(indices)
    m = np.median(ibis)
    #print(m)
    tempo = 1/m*60

    while tempo < 60:
        tempo *=2
        m/=2
    while tempo > 200:
        tempo /= 2
        m*=2
    
    print(f"tempo = {tempo}")
    
    add_values = []
    remove_indices = []

    for i, ibi in enumerate(ibis[:-1]):
        if abs(ibi-2*m) < eps: # 1 missing beat
            add_values.append(indices[i]+ibi/2)
        
        if abs(ibi-3*m) < eps: # 2 missing beats
            add_values.append(indices[i]+ibi/3)
            add_values.append(indices[i]+2/3*ibi)
        
        if abs(ibi+ibis[i+1]-m) < eps: # too many beats
            add_values.append(indices[i]+ibi/2)
            remove_indices.append(i+1)

    processed_indices = [idx for i, idx in enumerate(indices) if i not in remove_indices]
    processed_indices += add_values
    #print(f"added {len(add_values)}, removed {len(remove_indices)}")
    #input()
    return np.sort(processed_indices)


def topk(data, k=4):
    uniques, counts = np.unique(np.round(np.diff(data), decimals=6), return_counts=True)
    topk_idx = np.argsort(counts)[::-1][:k]
    topk = uniques[topk_idx]
    return topk

def similarity_score(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Pad the shorter array with zeros to match lengths
    if len(a) < len(b):
        a = np.pad(a, (0, len(b) - len(a)), 'constant')
    elif len(b) < len(a):
        b = np.pad(b, (0, len(a) - len(b)), 'constant')
    
    # Normalize the arrays
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0  # avoid division by zero if one array is constant

    a_norm = (a - np.mean(a)) / np.std(a)
    b_norm = (b - np.mean(b)) / np.std(b)
    
    # Compute cosine similarity (dot product of normalized vectors)
    similarity = np.dot(a_norm, b_norm) / len(a)
    
    # Scale from [-1, 1] to [0, 1]
    score = (similarity + 1) / 2
    return score

def overlap_score(candidates, history, decimals=3):
    cop1 = np.round(candidates.copy(), decimals=decimals)
    cop2 = np.round(history.copy(), decimals=decimals)
    return len(np.intersect1d(cop1, cop2))
    

def multi_agent_post(candidates):
    """
    get predictions for beat
    """
    startup_period = min(max(len(candidates)//3, 3), 10)
    lb = 20
    ub = 150
    
    # Get interval hypotheses
    intervals = topk(candidates, k=7) # interval hypotheses
    tempo = [60/i for i in intervals]
    for i, temp in enumerate(tempo):
        while tempo[i] > ub:
            tempo[i] = tempo[i]/2
        if temp < lb:
            tempo[i] = temp*2
        if tempo[i] >= 60:
            tempo.append(tempo[i]/2)
    if len(tempo)==0:
        tempo = [120]
    #print(f"Tempo Hypotheses: {tempo}")

    # Initialization

    agents = []
    for t_i in tempo:
        if t_i > 0:
            for e_j in candidates[:startup_period]:
                agent = beat_agent(beat_interval=60/t_i, 
                                pred = e_j+60/t_i, 
                                hist = [e_j], 
                                score = 0,
                                ibi=[])
                agents.append(agent)
    #print(f"Initialized {len(agents)} agents.")
    if len(agents) ==0:
        print(f"Tempi: {tempo}")
        print(f"Onset len: {len(candidates)}")

    # Main loop
    
    for i, a_i in enumerate(agents):
        while a_i.history[-1]<candidates[-1]:
            a_i.history.append(a_i.prediction)
            a_i.prediction += a_i.beatInterval
            a_i.score = overlap_score(candidates, a_i.history)
        
    #print(f"Choose best agent from {len(agents)} candidates")
    best_agent = choose_best_agent(agents, min_=False)

    #print(f"Beats: {best_agent.history}")
    predicted_beats = best_agent.history
    #predicted_beats = np.round(np.array(predicted_beats), decimals=5)
    #predicted_beats = np.sort(np.unique(predicted_beats))
    return predicted_beats