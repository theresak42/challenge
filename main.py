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

    data, labels, sr, hoplength = load_data(indir,methode="onsets", use_extra=False, fps=fps)
    
    print("Data loaded")

    len_data = int(len(data)*data_split)
    complete_dataset = AudioDataset(data, labels)
    train_dataset = AudioDataset(data[:len_data], labels[:len_data])
    val_dataset = AudioDataset(data[len_data:], labels[len_data:])

    

    train_dataloader = DataLoader(train_dataset, 4, shuffle= True, collate_fn = collate_fn)
    val_dataloader = DataLoader(val_dataset, 4, shuffle= False, collate_fn = collate_fn)
    print("Dataloader ready")

    tempo_data, tempos, *_ = load_data(indir, methode="tempo", use_extra=False, fps = fps)
    tempo_dataset = AudioDataset(tempo_data, tempos)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    model = CNN_model2().to(device)

    num_epochs = 35

    optimizer = AdamW(model.parameters(), lr = 0.0001)

    loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([24.0]).to(device))


    #train(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs, device=device, save_model=True, model_name="cnn3")

    model_path = os.getcwd()+r"\trained_models\cnn2.pt"

    #evaluate onsets
    #f1 = eval_o(model_path,complete_dataset,CNN_model2,fps,sr,hoplength, threshold=0.4)
    #print(f1)

    #evaluate tempos

    p_score = eval_t(model_path, tempo_dataset, CNN_model2)
    print(p_score)


    #get predictions
    
    #test_data, sr, hoplength = load_data(indir, train=False, fps=fps)

    #predict_o(model_path, test_data,CNN_model2, fps, r"pred\cnn_3.1.onset.pr")
    #predict_t(model_path, test_data, CNN_model2, r"pred\cnn_1.tempo.pr")
    


if __name__ == "__main__":
    #run program
    main(os.getcwd()+r"\data")