from pathlib import Path
import os
from torch.utils.data import Dataset
from scipy.io import wavfile
import numpy as np
import tqdm
import librosa


class AudioDataset(Dataset):
    def __init__(self, data, labels):
        super(AudioDataset, self).__init__()

        self.data = data
        self.labels = labels        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]


if __name__ == "__main__":
    pass


def preprocess_audio(filename, fps = 70):
    """
    loads file and transforms it to logmelspectogram
    fsp ... frames per seconds
    """
    sample_rate, signal = wavfile.read(filename)

    # convert from integer to float
    if signal.dtype.kind == 'i':
        signal = signal / np.iinfo(signal.dtype).max

    # convert from stereo to mono (just in case)
    if signal.ndim == 2:
        signal = signal.mean(axis=-1)
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

    return melspect, sample_rate, hop_length


def preprocess_labels(filename, methode, melspec=None, sr=None, hoplength=None):
    """
    preprocess labels, for different tasks
    onsets: transform timesteps to frames vector with onsets being 1 and rest being 0s
    tempo: read values
    beats: read values
    """
    
    if methode == "onsets":
        labels = []
        with open(filename, "r") as f:
            for line in f.readlines():
                values = list(map(float, line.replace("\n","").split(" ")))
                labels += values

        #converting to array of 0s and 1s
        final_labels = np.zeros(melspec.shape[1], dtype=np.float32)
        frame_times = librosa.frames_to_time(np.arange(melspec.shape[1]), sr=sr, hop_length=hoplength)
        for label in labels:
            index = np.argmin(np.abs(frame_times-label))
            final_labels[index] = 1.0
    
    elif methode == "tempo":
        with open(filename, "r") as f:
            final_labels = list(map(float, f.readline().replace("\n","").split("\t")))
        
    elif methode == "beats":
        labels = []
        with open(filename, "r") as f:
            for line in f.readlines():
                values = list(map(float, line.replace("\n","").split("\t")[0].split(" ")))
                labels += values

        #converting to array of 0s and 1s
        final_labels = np.zeros(melspec.shape[1], dtype=np.float32)
        frame_times = librosa.frames_to_time(np.arange(melspec.shape[1]), sr=sr, hop_length=hoplength)
        for label in labels:
            index = np.argmin(np.abs(frame_times-label))
            final_labels[index] = 1.0
        """
    elif methode == "beats":
        final_labels = []
        with open(filename, "r") as f:
            for line in f.readlines():
                try:
                    values = list(map(float, line.replace("\n","").split("\t")[0].split(" ")))
                except:
                    print(line.replace("\n","").split("\t")[0])
                final_labels += values
        """
    return final_labels
        


def load_data(indir, methode = "onsets", train=True, use_extra = True, fps=70):
    """
    loading the data
    methode:      which data to load [onsets, beats, tempo]
    train:        choose if to load train data(with labels) or test data used for predictions
    use_extra:    choose if to use extra data provided
    """
    indir = Path(indir)
    if train:
        match methode:
            case "onsets":
                suffix = ".onsets.gt"
                extra_dir = "train_extra_onsets"
            case "beats":
                suffix = ".beats.gt"
                extra_dir = "train_extra_tempobeats"
            case "tempo":
                suffix = ".tempo.gt"
                extra_dir = "train_extra_tempobeats"
            case _:
                raise KeyError("Needs to be one of: onsets, beats, tempo")

        #get all filenames
        new_indir = Path(os.path.join(indir,"train"))
        in_audio_files = list(new_indir.glob('*.wav'))    
        if use_extra:
            new_indir = Path(os.path.join(indir,extra_dir))
            in_audio_files += list(new_indir.glob('*.wav'))   
        infiles = tqdm.tqdm(in_audio_files, desc='File')


        #preprocess data
        data, labels = [], []
        for filename in infiles:
            melspec, sr,hoplength = preprocess_audio(filename, fps)

            data.append(melspec)
            label_filename = Path(filename).with_suffix(suffix)
            if methode == "onsets":
                pre_labels = preprocess_labels(label_filename, methode, melspec, sr, hoplength)
            elif methode == "tempo":
                pre_labels = preprocess_labels(label_filename, methode)
            elif methode == "beats":
                pre_labels = preprocess_labels(label_filename, methode, melspec, sr, hoplength)
                    
            labels.append(pre_labels)
            
        return data, labels, sr, hoplength

    else:
        #get data for predictions
        indir = Path(os.path.join(indir,"test"))
        in_audio_files = list(indir.glob('*.wav'))
        infiles = tqdm.tqdm(in_audio_files, desc='File')
        data= []
        for filename in infiles:
            melspec, sr,hoplength = preprocess_audio(filename, fps)
            data.append((melspec,filename))

        return data, sr, hoplength



#used for testing
if __name__=="__main__":
    data, labels = load_data(r"C:\Users\Jakob S\AI-Studium\6 Semester\Audio_and_Music_Processing\challenge\data", use_extra=False)
    dataset = AudioDataset(data, labels)