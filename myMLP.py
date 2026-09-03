import torch
import torch.nn as nn
from typing import List, Callable

class MyMLP(nn.Module):
    '''Custom MLP model supporting customizable layer depths and activation functions.'''
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int],
                 activation: Callable):
        super(MyMLP, self).__init__()
        self.flatten = nn.Flatten()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.net(x)