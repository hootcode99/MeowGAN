import torch
import torch.nn as nn


class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        out = self.model(x)

        return out