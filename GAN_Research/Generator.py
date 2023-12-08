import torch
from torch import nn


class SimpleConvGen(nn.Module):

    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(

            nn.BatchNorm1d(100), #BatchNorm

            nn.Conv1d(100, 200, kernel_size=1), #Convolutional Neural Networks

            nn.LeakyReLU(0.2, inplace=True), #LeakyReLU

            nn.BatchNorm1d(200),

            nn.Conv1d(200, 300, kernel_size=1),

            nn.LeakyReLU(0.2, inplace=True),

            nn.BatchNorm1d(300),

            nn.ConvTranspose1d(300, 200, kernel_size=1), #ConvTranspose

            nn.ReLU(),

            nn.BatchNorm1d(200),

            nn.ConvTranspose1d(200, 100, kernel_size=1),

            nn.Tanh() #Tanh
        )
    def forward(self, x):
        x = x.view(-1,100,7)

        #x = torch.transpose(x, 1, 2)  # Transposes the index values of the rows and columns of an array, similar to the transpose of a matrix.
        gen_out = self.gen(x)
        #gen_out = gen_out.view(-1,10, 10, 1)
        return gen_out