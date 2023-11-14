import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, filters, img_chnls):
        super().__init__()

        self.model = nn.Sequential(

            # IN: Noise Vector Dimensions
            nn.ConvTranspose2d(img_chnls, filters * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(filters * 8),
            nn.ReLU(True),

            # (filters * 8) x 4 x 4
            nn.ConvTranspose2d(filters * 8, filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filters * 4),
            nn.ReLU(True),

            # (filters * 4) x 8 x 8``
            nn.ConvTranspose2d(filters * 4, filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filters * 2),
            nn.ReLU(True),

            # (filters * 2) x 16 x 16``
            nn.ConvTranspose2d(filters * 2, filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),

            # filters x 32 x 32``
            nn.ConvTranspose2d(filters, img_chnls, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()

            # OUT: img_chnls x 64 x 64``
        )

    def forward(self, x):z
        return self.model(x)
