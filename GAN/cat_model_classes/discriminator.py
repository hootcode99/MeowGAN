import torch.nn as nn


class Disciminator(nn.Module):
    def __init__(self, filters, img_chnls):
        super().__init__()

        self.model = nn.Sequential(

            # IN: img_chnls x 64 x 64
            nn.utils.spectral_norm(nn.Conv2d(img_chnls, filters, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),

            # filters x 32 x 32
            nn.utils.spectral_norm(nn.Conv2d(filters, filters * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),

            # (filters * 2) x 16 x 16
            nn.utils.spectral_norm(nn.Conv2d(filters * 2, filters * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),

            # (filters * 4) x 8 x 8
            nn.utils.spectral_norm(nn.Conv2d(filters * 4, filters * 8, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),

            # (filters * 8) x 4 x 4
            nn.utils.spectral_norm(nn.Conv2d(filters * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)),
            nn.Sigmoid()

            # OUT: 1 x 1 x 1
        )

    def forward(self, x):
        return self.model(x)
