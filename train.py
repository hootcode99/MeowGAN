import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as v2
import torchvision.utils
import utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data.cat_dataclass import CatDataset
from models.generator import Generator
from models.discriminator import Disciminator

# Hyperparameters -----------------------------------------------------------------
EPOCHS = 20
LEARNING_RATE = .001
BATCH_SIZE = 128
NOISE_DIM = 128
NUM_FILTERS = 32
IMAGE_SIZE = 64
IMG_CHNLS = 3
SEED = 42

# Torch Setup ---------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()
torch.manual_seed(SEED)
np.random.seed(SEED)

# Data Loading --------------------------------------------------------------------
transforms = v2.Compose([
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.ToTensor(),
])

cat_faces = CatDataset(path='data/dataset/', transforms=transforms)

train_loader = DataLoader(cat_faces, batch_size=BATCH_SIZE)

# Model Setup ---------------------------------------------------------------------
generator = Generator(NUM_FILTERS, NOISE_DIM, IMG_CHNLS).to(device)
discriminator = Disciminator(NUM_FILTERS, IMG_CHNLS).to(device)

# Initialize Weights using a normal distribution
generator.apply(utils.init_weights)
discriminator.apply(utils.init_weights)

gen_optimizer = optim.Adam(generator.parameters(), LEARNING_RATE, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), LEARNING_RATE, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# Training Loop ---------------------------------------------------------------------
fake_data = None

# epoch_gen_losses = np.array([])
# epoch_disc_losses = np.array([])

for epoch in range(EPOCHS):
    # reset loss records
    batch_gen_losses = np.array([])
    batch_disc_losses = np.array([])

    for idx, batch in enumerate(train_loader, 0):
        # Train discriminator on batch of real data
        discriminator.zero_grad()

        real_data = batch.to(device)
        batch_size = batch.size(0)

        real_labels = torch.ones(batch_size, dtype=torch.float, device=device)
        real_outputs = discriminator(real_data).reshape(-1)
        real_loss = criterion(real_outputs, real_labels)
        real_loss.backward()

        # D_x = real_outputs.mean().item()

        # Train discriminator on batch of fake data
        latent_noise = torch.randn(batch_size, NOISE_DIM, 1, 1, device=device)

        fake_data = generator(latent_noise)
        fake_labels = torch.zeros(batch_size, dtype=torch.float, device=device)
        fake_outputs = discriminator(fake_data.detach()).reshape(-1)
        fake_loss = criterion(fake_outputs, fake_labels)
        fake_loss.backward()

        # D_G_z = fake_outputs.mean().item()

        disc_loss_sum = real_loss + fake_loss
        disc_optimizer.step()

        # Train generator
        generator.zero_grad()
        fake_outputs = discriminator(fake_data).reshape(-1)
        gen_loss = criterion(fake_outputs, real_labels)
        gen_loss.backward()

        # D_G_z2 = fake_outputs.mean().item()
        gen_optimizer.step()

        # Show progress in terminal
        if idx % 50 == 0:
            print('[%d/%d][%d/%d]\tDisc_Loss: %.4f\tGen_Loss: %.4f' %
                  (epoch, EPOCHS, idx, len(train_loader), disc_loss_sum.item(), gen_loss.item()))

        # Optional, More Detailed Terminal Readout
        # if idx % 50 == 0:
        #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' %
        #           (epoch, EPOCHS, idx, len(train_loader), disc_loss_sum.item(), gen_loss.item(), D_x, D_G_z, D_G_z2))

        # Add batch losses to record
        batch_gen_losses = np.append(batch_gen_losses, gen_loss.item())
        batch_disc_losses = np.append(batch_disc_losses, disc_loss_sum.item())

    # Add epoch losses to record
    epoch_gen_loss = batch_gen_losses.mean().item()
    epoch_disc_loss = batch_disc_losses.mean().item()
    # epoch_gen_losses = np.append(epoch_gen_losses, epoch_gen_loss)
    # epoch_disc_losses = np.append(epoch_disc_losses, epoch_disc_loss)
    writer.add_scalar(tag="Epoch Gen Loss", scalar_value=epoch_gen_loss, global_step=epoch)
    writer.add_scalar(tag="Epoch Disc Loss", scalar_value=epoch_disc_loss, global_step=epoch)

    # Add Generator output images to tensorboard
    generator_grid = torchvision.utils.make_grid(fake_data[0:18], 6, 2)
    writer.add_image(tag="gen_images", img_tensor=generator_grid, global_step=epoch)

# save models
writer.close()
