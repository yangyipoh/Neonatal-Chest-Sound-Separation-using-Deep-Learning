import torch
from torch import nn


class DeepEncoder(nn.Module):
    def __init__(self, in_channels:int=1, out_channels:int=512, kernel_size:int=3, num_layers:int=7) -> None:
        super().__init__()
        layers = [DeepEncoderBlock(in_channels, out_channels, kernel_size=kernel_size)]
        layers += [DeepEncoderBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        ) for _ in range(num_layers-1)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DeepEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same', stride=kernel_size//2)
        self.normalization = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x:torch.Tensor):
        residual = x
        x = self.conv(x)
        x = self.normalization(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.activation(x)
        return x + residual
