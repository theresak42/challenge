from pathlib import Path
import os

class DataLoader():
    def __init__(self, indir, methode = "onsets", train=True, use_extra = True):
        indir = Path(indir)

        if train:
            if not use_extra:
                indir = Path(os.path.join(indir,"train"))
            in_audio_files = list(indir.glob('*.wav'))
            match methode:
                case "onsets":
                    pass
                case "beats":
                    pass
                case "tempo":
                    pass
                case _:
                    raise KeyError("Needs to be one of: onsets, beats, tempo")
            
        
        else:
            indir = Path(os.path.join(indir,"test"))
            in_audio_files = list(indir.glob('*.wav'))
        
        pass



if __name__ == "__main__":

    test = DataLoader(r"C:\Users\Jakob S\AI-Studium\6 Semester\Audio_and_Music_Processing\challenge\data", use_extra=False)

