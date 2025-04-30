from data import *
from model import *
from utils import *
from train import train

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


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

    model = CNN_model().to(device)

    num_epochs = 20

    optimizer = AdamW(model.parameters(), lr = 0.0001)

    loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([30.0]).to(device))


    #train(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs, device=device, save_model=True, model_name="friedrich")


    f1 = eval(r"C:\Users\Jakob S\AI-Studium\6 Semester\Audio_and_Music_Processing\challenge\trained_models\friedrich.pt",complete_dataset,fps,sr,hoplength, device )
    print(f1)
    





if __name__ == "__main__":
    main(r"C:\Users\Jakob S\AI-Studium\6 Semester\Audio_and_Music_Processing\challenge\data")