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

def preprocess_labels(filename, methode, melspec, sr, hoplength):
    labels = []
    with open(filename, "r") as f:
        for line in f.readlines():
            values = list(map(float, line.replace("\n","").split(" ")))
            labels += values

    if methode == "onsets":
        binary_labels = np.zeros(melspec.shape[1], dtype=np.float32)
        frame_times = librosa.frames_to_time(np.arange(melspec.shape[1]), sr=sr, hop_length=hoplength)
        for label in labels:
            index = np.argmin(np.abs(frame_times-label))
            binary_labels[index] = 1.0
    
    return binary_labels



def load_data(indir, methode = "onsets", train=True, use_extra = True, fps=70):
    indir = Path(indir)
    #print(indir)
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

        new_indir = Path(os.path.join(indir,"train"))
        in_audio_files = list(new_indir.glob('*.wav'))    
        if use_extra:
            new_indir = Path(os.path.join(indir,extra_dir))
            in_audio_files += list(new_indir.glob('*.wav'))   

        
            
        infiles = tqdm.tqdm(in_audio_files, desc='File')
        data, labels = [], []
        for filename in infiles:
            melspec, sr,hoplength = preprocess_audio(filename, fps)
            #print(melspec.shape)
            data.append(melspec)
            label_filename = Path(filename).with_suffix(suffix)
            preproc_labels = preprocess_labels(label_filename, methode, melspec, sr, hoplength)
            #print(len(preproc_labels))
            labels.append(preproc_labels)
            
        return data, labels, sr, hoplength

    else:
        indir = Path(os.path.join(indir,"test"))
        in_audio_files = list(indir.glob('*.wav'))
        infiles = tqdm.tqdm(in_audio_files, desc='File')
        data= []
        for filename in infiles:
            melspec, sr,hoplength = preprocess_audio(filename, fps)
            data.append((melspec,filename))

        return data, sr, hoplength

#TODO: align files to labels
#TODO: dont need audio files if we dont have labels for it

if __name__=="__main__":
    data, labels = load_data(r"C:\Users\Jakob S\AI-Studium\6 Semester\Audio_and_Music_Processing\challenge\data", use_extra=False)
    dataset = AudioDataset(data, labels)