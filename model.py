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


if __name__ == "__main__":
    test_random = torch.rand((2,1,80,1200))
    a = CNN_model()
    print(a(test_random).shape)
