# ========================================
#  @project: dcgan-from-scratch
#  @filename: ~/Developer/dcgan-from-scratch/trainer/model.py
# ========================================
#  @author: @github/sounishnath003
#  @generatedID: 129d73ca-90da-4170-a9c4-3dc2ac8d34fc
#  @createdAt: 27.01.2024 +05:30
# ========================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, n_channels: int, feature_dim: int, *args, **kwargs) -> None:
        super(Discriminator, self).__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=feature_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=feature_dim,
                out_channels=feature_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=feature_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=feature_dim * 2,
                out_channels=feature_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=feature_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=feature_dim * 4,
                out_channels=feature_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=feature_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=feature_dim * 8,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=0,
            ),
            nn.Sigmoid(),
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
        return self.net.forward(x)


class Generator(nn.Module):
    def __init__(
        self, channels_noise: int, channels_img: int, feature_dim: int, *args, **kwargs
    ) -> None:
        super(Generator, self).__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=channels_noise,
                out_channels=feature_dim * 16,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(feature_dim * 16),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=feature_dim * 16,
                out_channels=feature_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(feature_dim * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=feature_dim * 8,
                out_channels=feature_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=feature_dim * 4,
                out_channels=feature_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=feature_dim * 2,
                out_channels=channels_img,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
        return self.net.forward(x)
