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

    data, labels, sr, hop = load_data(indir, methode="beats", train=True, use_extra=True)
    beats_dataset = AudioDataset(data, labels)

    
    # calculate 0-1 ratio
    """
    ratios = []
    for d, l in beats_dataset:
        ratio = len(np.argwhere(l==1))/len(l)
        ratios.append(ratio)
    print(f"Ratios: {ratios[:10]}")
    mean_ratio = np.mean(np.array(ratios))
    print(mean_ratio)

    raise(NotImplementedError())
    # result: 0.028820094705963833 ~ 1:35
    """
    
    # multi-agent beat detection
    """
    len_data = int(len(data)*data_split)
    complete_dataset = AudioDataset(data, labels)
    train_dataset = AudioDataset(data[:len_data], labels[:len_data])
    val_dataset = AudioDataset(data[len_data:], labels[len_data:])

    tempo_data, tempo_labels, _, _ = load_data(indir, methode="tempo", train=True, use_extra=False)
    tempo_dataset = AudioDataset(tempo_data, tempo_labels)

    beats_data, beats_labels, _, _ = load_data(indir, methode="beats", train=True, use_extra=False)
    beats_dataset = AudioDataset(beats_data, beats_labels)
    
    model_path = os.getcwd()+r"\trained_models\cnn2.pt"

    onsets = get_onsets(model_path, complete_dataset, CNN_model2, fps,sr,hop, threshold=0.4)
    tempo = get_tempo(model_path, tempo_dataset, CNN_model2)
    b_score = eval_b(beats_dataset, onsets, tempo)

    print(f"Run successfull.")
    print(f"b_score = {b_score}")
    return
    # after disabling this part, you need to enable the correct preprocess_labels else-branch in data.py 
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

    num_epochs = 20

    optimizer = AdamW(model.parameters(), lr = 0.0001)

    loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([30.0]).to(device))


    train(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs, device=device, save_model=True, model_name="beats_cnn4")

    model_path = os.getcwd()+r"\trained_models\beats_cnn4.pt"
    score = eval_o(model_path, beats_dataset, model_class=CNN_model2, odf_rate=fps, sr=sr, hoplength=hop, threshold=0.4, type_="beats")
    print(score)
    
    # TODO: test different thresholds
    # load model, evaluate, etc.

if __name__ == "__main__":
    #run program
    main(os.getcwd()+r"\data")