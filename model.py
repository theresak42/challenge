from torch import nn
import torch

class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2,1)),

            nn.Conv2d(32,64,kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(4,1)),

            nn.Conv2d(64,128,kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2,1)),

            nn.Conv2d(128,1,kernel_size=(5,1), padding=(0,0))
        )
        """nn.Conv2d(64,128,kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(4,1)),"""

    def forward(self, x):
        x = self.CNN(x)
        x = x.squeeze(2).squeeze(1)
        return x


class CNN_model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=(3,5), padding="same", padding_mode="replicate"),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2,1)),

            nn.Conv2d(32,128,kernel_size=(3,5), padding="same", padding_mode="replicate"),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(4,1)),

            nn.Conv2d(128,256,kernel_size=(3,5), padding="same", padding_mode="replicate"),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2,1)),

            nn.Conv2d(256,1,kernel_size=(5,1), padding=(0,0))
        )

    def forward(self, x):
        x = self.CNN(x)
        x = x.squeeze(2).squeeze(1)
        return x




class RNN_model(nn.Module):
    def __init__(self, input_size=80, hidden_size=256, num_layers=4, bidirectional=True):
        super().__init__()

        self.lin = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.LayerNorm(input_size)
        )

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1) 

    def forward(self, x):
        x = x.squeeze(1) 
        x = x.permute(0, 2, 1)  # (Batch, Time, Frequency)

        x = self.lin(x)
        rnnout, _ = self.rnn(x)
        out = self.fc(rnnout).squeeze(-1)

        return out

if __name__ == "__main__":
    test_random = torch.rand((2,1,80,1200))
    a = RNN_model()

    print(a(test_random).shape)
