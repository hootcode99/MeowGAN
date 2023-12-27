import torch
import torch.nn as nn
import torchvision.transforms as v2


# Initialize Layer Weights using a Gaussian Distribution
def init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        torch.nn.init.normal_(layer.weight, 0.0, 0.02)
    elif isinstance(layer, nn.ConvTranspose2d):
        torch.nn.init.normal_(layer.weight, 0.0, 0.02)
    elif isinstance(layer, nn.BatchNorm2d):
        torch.nn.init.normal_(layer.weight, 1.0, 0.02)


# Inverse of (-1, 1) Normalization
def denormalize(norm_imgs):
    invTrans = v2.Compose([
        v2.Normalize(mean=[0., 0., 0.], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
        v2.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.]),
    ])

    denorm_imgs = invTrans(norm_imgs)

    return denorm_imgs


# Apply Gaussian Noise to a Batch of Images
def gaussian_noise(imgs, mean, stddev):
    with torch.no_grad():
        noise = torch.randn_like(imgs) * stddev + mean
        return imgs + noise
