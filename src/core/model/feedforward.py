import torch
from torch import nn, functional

from enum import Enum

class ActivationTypes(Enum):
    RELU = 'relu',
    GELU = 'gelu'

class FeedForward(nn.Module):
    def __init__(self, layers=[64, 128, 128, 64], activation: ActivationTypes=ActivationTypes.GELU, dropout=0.1):
        super().__init__()
        
        self.ff_layers = nn.ModuleList([nn.Linear(p, c) for p, c in zip(layers, layers[1:])])
        self.activation = nn.GELU() if activation is ActivationTypes.GELU else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.ff_layers:
            X = self.activation(layer(X))
        
        return self.dropout(X)