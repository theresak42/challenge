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

    data, labels, sr, hop = load_data(indir, methode="beats", train=True, use_extra=False)
    beats_dataset = AudioDataset(data, labels)


    """
    # multi-agent beat detection
    onsets = get_onsets(model_path, complete_dataset, CNN_model2, fps,sr,hoplength, threshold=0.4)
    tempo = get_tempo(model_path, tempo_dataset, CNN_model2)
    b_score = eval_b(beats_dataset, onsets, tempo)

    print(f"Run successfull.")
    print(f"b_score = {b_score}")
    """

    
    print("Data loaded")
    len_data = int(len(data)*data_split)
    train_dataset = AudioDataset(data[:len_data], labels[:len_data])
    val_dataset = AudioDataset(data[len_data:], labels[len_data:])

    

    train_dataloader = DataLoader(train_dataset, 4, shuffle= True, collate_fn = collate_fn)
    val_dataloader = DataLoader(val_dataset, 4, shuffle= False, collate_fn = collate_fn)
    print("Dataloader ready")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    model = CNN_model2().to(device)

    num_epochs = 1

    optimizer = AdamW(model.parameters(), lr = 0.0001)

    loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]).to(device))


    train(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs, device=device, save_model=True, model_name="beats_cnn1")

    model_path = os.getcwd()+r"\trained_models\beats_cnn1.pt"
    score = eval_o(model_path, beats_dataset, model_class=CNN_model2, odf_rate=fps, sr=sr, hoplength=hop, threshold=0.8, type_="beats")
    print(score)

if __name__ == "__main__":
    #run program
    main(os.getcwd()+r"\data")