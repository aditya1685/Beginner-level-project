# making model.py
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels= 64,kernel_size=3,stride= 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64,out_channels= 64,kernel_size=3,stride= 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2))
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels= 128,kernel_size=3,stride= 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128,out_channels= 128,kernel_size=3,stride= 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2))
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels = 128,out_channels= 256,kernel_size=3,stride= 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256,out_channels= 256,kernel_size=3,stride= 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2))
        )
        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels = 256,out_channels= 512,kernel_size=3,stride= 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512,out_channels= 512,kernel_size=3,stride= 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2))
        )
        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels = 512,out_channels= 512,kernel_size=3,stride= 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512,out_channels= 512,kernel_size=3,stride= 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 25088,
                     out_features = 4096),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features = 4096,
                     out_features = 4096),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features = 4096,
                     out_features = 3),
        )
    def forward(self,x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.classifier(x)
        return x