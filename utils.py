import torch
import torch.nn as nn


def init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        torch.nn.init.normal_(layer.weight, 0.0, 0.02)
    elif isinstance(layer, nn.ConvTranspose2d):
        torch.nn.init.normal_(layer.weight, 0.0, 0.02)
    elif  isinstance(layer, nn.BatchNorm2d):
        torch.nn.init.normal_(layer.weight, 1.0, 0.02)




