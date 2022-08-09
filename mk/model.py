import torch.nn as nn
from abc import ABC, abstractproperty


N_MOVES = 12


class AView(nn.Module):
    @abstractproperty
    def output_size(self) -> int:
        pass


class View(AView):
    @property
    def output_size(self) -> int:
        return 1680

    @staticmethod
    def conv_block(in_channels: int, out_channels: int):
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        activation = nn.ReLU()
        pooling = nn.MaxPool2d(2, 2)
        return nn.Sequential(conv, activation, pooling)

    def __init__(self, frames):
        super().__init__()
        self.conv1 = self.conv_block(frames, 12)
        self.conv2 = self.conv_block(12, 24)
        self.conv3 = self.conv_block(24, 36)
        self.conv4 = self.conv_block(36, 36)
        self.conv5 = self.conv_block(36, 24)
        self.flat = nn.Flatten()

    def forward(self, frames):
        # B H W C --> B C H W
        x = frames.permute(0, 3, 1, 2)  # 224x320x3
        x = self.conv1(x)  # 112x160
        x = self.conv2(x)  # 56x80x24
        x = self.conv3(x)  # 28x40x36
        x = self.conv4(x)  # 14x20x36
        x = self.conv5(x)  # 7x10x24
        x = self.flat(x)
        return x


class Actor(nn.Module):
    def __init__(self, view: AView, frames):
        super().__init__()
        self.view = view
        self.actions = nn.Linear(self.view.output_size, N_MOVES)

    def forward(self, frames):
        x = frames
        img_embeddings = self.view(x)
        actions = self.actions(img_embeddings)
        return actions


class HealthPredictor(nn.Module):
    def __init__(self, view: AView):
        super().__init__()
        self.view = view
        self.actions = nn.Linear(self.view.output_size, 1)

    def forward(self, frames):
        x = frames
        img_embeddings = self.view(x)
        health_predictions = self.actions(img_embeddings)
        return health_predictions
