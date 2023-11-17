import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as v2
import utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data.cat_dataclass import CatDataset
from models.generator import Generator
from models.discriminator import Disciminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fixed Hyperparameters
EPOCHS = 10
LEARNING_RATE = .001
BATCH_SIZE = 128
NOISE_DIM = 128
NUM_FILTERS = 32
IMAGE_SIZE = 64
IMG_CHNLS = 3

# Data Loading
transforms = v2.Compose([
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.ToTensor(),
    # v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

cat_faces = CatDataset(path='data/dataset/', transforms=transforms)

train_loader = DataLoader(cat_faces, batch_size=BATCH_SIZE)

# Model Setup
generator = Generator(NUM_FILTERS, NOISE_DIM, IMG_CHNLS).to(device)
discriminator = Disciminator(NUM_FILTERS, IMG_CHNLS).to(device)

# Initialize Weights using a normal distribution
generator.apply(utils.init_weights)
discriminator.apply(utils.init_weights)

gen_optimizer = optim.Adam(generator.parameters(), LEARNING_RATE, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), LEARNING_RATE, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# Training Loop
for epoch in range(EPOCHS):

    for idx, batch in enumerate(train_loader, 0):
        # Train Discriminator on batch of real data
        discriminator.zero_grad()

        real_data = batch.to(device)
        batch_size = batch.size(0)

        real_labels = torch.ones(batch_size, dtype=torch.float, device=device)
        real_outputs = discriminator(real_data).reshape(-1)
        real_loss = criterion(real_outputs, real_labels)
        real_loss.backward()

        D_x = real_outputs.mean().item()

        # Train Discriminator on batch of fake data
        latent_noise = torch.randn(batch_size, NOISE_DIM, 1, 1, device=device)

        fake_data = generator(latent_noise)
        fake_labels = torch.zeros(batch_size, dtype=torch.float, device=device)
        fake_outputs = discriminator(fake_data.detach()).reshape(-1)
        fake_loss = criterion(fake_outputs, fake_labels)
        fake_loss.backward()

        D_G_z = fake_outputs.mean().item()

        disc_error = real_loss + fake_loss
        disc_optimizer.step()

        # Train Generator
        generator.zero_grad()
        fake_outputs = discriminator(fake_data).reshape(-1)
        gen_loss = criterion(fake_outputs, real_labels)
        gen_loss.backward()

        D_G_z2 = fake_outputs.mean().item()
        gen_optimizer.step()

