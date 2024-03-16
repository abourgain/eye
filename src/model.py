"""
Implement AlexNet Architecture
"""

import torch
from torch import nn


class AlexNet(nn.Module):
    """
    AlexNet Architecture
    """

    def __init__(self, in_channels: int = 3, output_dim: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.output_dim = output_dim

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(6400, 4096), nn.ReLU())
        self.fc_2 = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU())
        self.fc_3 = nn.Sequential(nn.Linear(4096, output_dim))

    def forward(self, x):
        """
        Forward pass
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.flatten(out)
        out = self.fc_1(out)
        out = self.fc_2(out)
        out = self.fc_3(out)
        return out


def test():
    """
    Test AlexNet model
    """
    x = torch.randn((3, 3, 224, 224))
    model = AlexNet(in_channels=3, output_dim=10)
    preds = model(x)
    print(f"Shape of x: {x.shape}")
    print(f"Shape of preds: {preds.shape}")
    assert preds.shape[0] == x.shape[0]
    assert preds.shape[1] == 10


if __name__ == "__main__":
    test()
