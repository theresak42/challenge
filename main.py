from data import *
from model import *
from utils import *
from train import train

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss

def main(indir):
    data_split = 0.8
    fps =70

    data, labels, sr, hoplength = load_data(indir, use_extra=True, fps=fps)
    print("Data loaded")

    len_data = int(len(data)*data_split)
    complete_dataset = AudioDataset(data, labels)
    train_dataset = AudioDataset(data[:len_data], labels[:len_data])
    val_dataset = AudioDataset(data[len_data:], labels[len_data:])

    train_dataloader = DataLoader(train_dataset, 4, shuffle= True, collate_fn = collate_fn)
    val_dataloader = DataLoader(val_dataset, 4, shuffle= False, collate_fn = collate_fn)
    print("Dataloader ready")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    model = RNN_model().to(device)

    num_epochs = 30

    optimizer = AdamW(model.parameters(), lr = 0.0001)

    loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([24.0]).to(device))


    #train(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs, device=device, save_model=True, model_name="rnn1")

    #model_path = r"C:\Users\Jakob S\AI-Studium\6 Semester\Audio_and_Music_Processing\challenge\trained_models\friedrich.pt"
    #f1-score 0.833941558133436
    #train 20 epochs, cnn1

    #model_path = r"C:\Users\Jakob S\AI-Studium\6 Semester\Audio_and_Music_Processing\challenge\trained_models\cnn2.pt"
    #f1-score 0.8407248444261645
    #train 30 epochs, cnn2
    #best val loss 0.36096498

    model_path = r"C:\Users\Jakob S\AI-Studium\6 Semester\Audio_and_Music_Processing\challenge\trained_models\cnn2.pt"
    #0.786455088685055   hidden 128 numlayer 2

    #f1 = eval(model_path,complete_dataset,CNN_model2,fps,sr,hoplength, threshold=0.4)
    #print(f1)


    test_data, sr, hoplength = load_data(indir, train=False, fps=fps)

    predict(model_path, test_data,CNN_model2, fps, r"pred\cnn_3.1.onset.pr")
    
    #Note cnn_3.onset.pr was produced with custom_detect_peaks2(pred, threshold=0.5)
    #TODO: if CNN_4 performs worse, can also try custom_detect_peaks2() with threshold 0.4, since it performed better


    #TODO:
    #custom peak detection function: threshold 0.5, idea threshold falls with iterations, adaptive thresholding





if __name__ == "__main__":
    main(r"C:\Users\Jakob S\AI-Studium\6 Semester\Audio_and_Music_Processing\challenge\data")