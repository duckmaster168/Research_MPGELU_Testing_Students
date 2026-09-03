import torch
import torch.nn as nn
from typing import Iterable, Callable

class MyCNN(nn.Module):
    '''Custom CNN model with flexible architecture[cite: 10]'''
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 activation: Callable,
                 params: Iterable):
        super(MyCNN, self).__init__()
        self.input_shape = input_shape[cite: 10]
        self.activation = activation[cite: 10]
        self.conv_layers = self.convo_layers(params)[cite: 10]

        self.fc = nn.Sequential(
            nn.Linear(in_features=128 * 4 * 4, out_features=256),[cite: 10]
            self.activation(),[cite: 10]
            nn.Linear(in_features=256, out_features=256),[cite: 10]
            self.activation(),[cite: 10]
            nn.Linear(in_features=256, out_features=output_shape)[cite: 10]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)[cite: 10]
        x = x.reshape(x.shape[0], -1)[cite: 10]
        x = self.fc(x)[cite: 10]
        return x
    
    def convo_layers(self, arch: Iterable) -> nn.Sequential:
        Mylayers = []
        for k in arch:[cite: 10]
            if type(k) == int:[cite: 10]
                output_channels = k[cite: 10]
                Mylayers += [
                    nn.Conv2d(self.input_shape, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),[cite: 10]
                    self.activation()[cite: 10]
                ]
                self.input_shape = k[cite: 10]
            elif k == "MaxPool":[cite: 10]
                Mylayers += [nn.MaxPool2d(kernel_size=(2, 2))][cite: 10]
        return nn.Sequential(*Mylayers)[cite: 10]
