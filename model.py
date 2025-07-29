# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, functional

def conv(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False, bn=True, pooling=True,
         act_layer=neuron.LIFNode):
    return nn.Sequential(
        layer.MaxPool2d(kernel_size=2, stride=2) if pooling else layer.SeqToANNContainer(nn.Sequential(nn.Identity())),
        layer.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
        layer.BatchNorm2d(out_channels) if bn else layer.SeqToANNContainer(nn.Sequential(nn.Identity())),
        act_layer(detach_reset=True),
        )

class SNN(nn.Module):
    def __init__(self, T:int, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.T = T
        self.spike_features = nn.Sequential(
            conv(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), pooling=False, bn=False),
            conv(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), pooling=True, bn=False),
            conv(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), pooling=True, bn=False),
            conv(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), pooling=False, bn=False),
            conv(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), pooling=False, bn=False),
            layer.MaxPool2d(kernel_size=3, stride=2),
        )
        self.spike_classifier = nn.Sequential(
            layer.Flatten(),
            layer.BatchNorm1d(256 * 6 * 6),
            layer.Linear(256 * 6 * 6, 1024, bias=True),
            neuron.LIFNode(detach_reset=True),
            layer.BatchNorm1d(1024),
            layer.Linear(1024, 1024, bias=True),
            neuron.LIFNode(detach_reset=True),
            # layer.BatchNorm1d(1024),
            layer.Linear(1024, num_classes, bias=True),
        )

        functional.set_step_mode(self, step_mode='m')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.spike_features(x)
        x = self.spike_classifier(x)
        x = x.mean(0)
        x = torch.softmax(x, dim=1)
        return x