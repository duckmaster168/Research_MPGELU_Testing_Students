import torch
import torch.nn as nn
from typing import Iterable, Callable

class MyCNN(nn.Module):
    """Custom CNN model supporting custom activation functions and gradient tracking[cite: 4]."""
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 activation: Callable,
                 params: Iterable):
        super(MyCNN, self).__init__()
        self.input_shape = input_shape
        self.activation = activation
        self.conv_layers = self.convo_layers(params)

        self.fc = nn.Sequential(
            nn.Linear(in_features=128 * 4 * 4, out_features=256),
            self.activation(),
            nn.Linear(in_features=256, out_features=256),
            self.activation(),
            nn.Linear(in_features=256, out_features=output_shape)
        )

    def convo_layers(self, arch):
        layers = []
        for k in arch:
            if isinstance(k, int):
                output_channels = k
                layers += [
                    nn.Conv2d(self.input_shape, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    self.activation()
                ]
                self.input_shape = k
            elif k == "MaxPool":
                layers += [nn.MaxPool2d(kernel_size=(2, 2))]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_layer_grad_norms(self):
        """Calculates gradient norms across weight layers."""
        grad_norms = {}
        for name, param in self.named_parameters():
            if param.grad is not None and "weight" in name:
                grad_norms[name] = param.grad.norm().item()
        return grad_norms

    def get_lambda_values(self):
        """Extracts active lambda values across custom GELU layers."""
        lambdas = []
        for module in self.modules():
            if hasattr(module, 'get_lambda'):
                lambdas.append(module.get_lambda())
        return lambdas
