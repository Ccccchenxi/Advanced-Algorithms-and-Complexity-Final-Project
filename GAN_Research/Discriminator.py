import torch
from torch import nn


class SimpleConvDisc(nn.Module):

    def __init__(self, input_features):
        super().__init__()

        self.input_features = input_features

        self.disc = nn.Sequential(

            nn.BatchNorm1d(self.input_features), #BatchNorm

            nn.Conv1d(self.input_features, 200, kernel_size=1), #Convolutional Neural Networks

            nn.LeakyReLU(0.2, inplace=True), #LeakyReLU

            nn.BatchNorm1d(200),

            nn.Conv1d(200, 300, kernel_size=1),

            nn.BatchNorm1d(300),

            nn.ConvTranspose1d(300, 200, kernel_size=1), #ConvTranspose

            nn.ReLU(),

            nn.BatchNorm1d(200),

            nn.ConvTranspose1d(200, 100, kernel_size=1),

            nn.ReLU(),

            nn.BatchNorm1d(100),

            nn.ConvTranspose1d(100, 1, kernel_size=1),

            nn.Sigmoid() #Sigmoid

        )

    def forward(self, x):
        x = x.view(-1,100,7)
        #x = torch.transpose(x, 1, 2)
        x = self.disc(x)
        #x = torch.transpose(x, 1, 2)
        return x