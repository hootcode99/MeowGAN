import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as v2
import torchvision.utils
from torch.cuda.amp import GradScaler
from utils import init_weights, denormalize, gaussian_noise
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data.cat_dataclass import CatDataset
from GAN.cat_model_classes.generator import Generator
from GAN.cat_model_classes.discriminator import Disciminator

# Hyperparameters ------------------------------------------------------------
EPOCHS = 100
LEARNING_RATE_GEN = .0002
LEARNING_RATE_DISC = .0003
BATCH_SIZE = 128
NUM_FILTERS_GEN = 64
NUM_FILTERS_DISC = 64
NOISE_DIM = 100
use_amp = True

# Static Parameters ---------------------------------------------------------------
IMAGE_DIM = 64
IMG_CHNLS = 3
SEED = 42

# Torch Setup ---------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)
print(f'Using Device: {device}\n')

model_save_path = f'./saved_models/cat_model/best_gen.pth'
dict_save_path = f'./saved_models/cat_state_dict/best_gen.pth'

torch.manual_seed(SEED)
np.random.seed(SEED)

name = (
        f'VANILLA GLR_{LEARNING_RATE_GEN} '
        f'DLR_{LEARNING_RATE_DISC} '
        f'BSZ_{BATCH_SIZE} '
        f'GFLT_{NUM_FILTERS_GEN} '
        f'DFLT_{NUM_FILTERS_DISC}'
        )

writer = SummaryWriter(log_dir=f"tensorboard_gan_cat/{name}")

# Data Loading --------------------------------------------------------------------
transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToTensor(),
    v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

cat_faces = CatDataset(path='../data/cat_faces/', transforms=transforms)

train_loader = DataLoader(cat_faces, batch_size=BATCH_SIZE, drop_last=True)

# Model Setup ---------------------------------------------------------------------
generator = Generator(NUM_FILTERS_GEN, NOISE_DIM, IMG_CHNLS).to(device)
discriminator = Disciminator(NUM_FILTERS_DISC, IMG_CHNLS).to(device)

# Initialize Weights using a Normal distribution
generator.apply(init_weights)
discriminator.apply(init_weights)

# Initialize Optimizers
gen_optimizer = optim.Adam(generator.parameters(), LEARNING_RATE_GEN, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), LEARNING_RATE_DISC, betas=(0.5, 0.999))

# Initialize Scheduler
# disc_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(disc_optimizer, T_0=10, eta_min=0.00005)
disc_scheduler = optim.lr_scheduler.CosineAnnealingLR(disc_optimizer, T_max=10, eta_min=0.00018)
# disc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(disc_optimizer, mode='min', factor=0.75, patience=10)

# Initialize Loss Function
loss_fn = nn.BCELoss()

# Initialize AMP Grad-Scaler
gen_scaler = GradScaler(enabled=use_amp)
disc_scaler = GradScaler(enabled=use_amp)

# Training Loop --------------------------------------------------------------------
if use_amp:
    print("Begining Training...")
    print("Using Mixed Precision\n")
else:
    print("Begining Training...\n")

for epoch in range(EPOCHS):
    batch_gen_losses = np.array([])
    batch_disc_losses = np.array([])

    for idx, batch in enumerate(train_loader, 0):
        discriminator.zero_grad(set_to_none=True)
        generator.zero_grad(set_to_none=True)

        real_data, batch_size = batch.to(device), batch.size(0)
        noisy_real_data = gaussian_noise(real_data, 0, 0.05)

        fake_labels = torch.zeros(batch_size, dtype=torch.float, device=device)
        real_labels = (1 - 0.8) * torch.rand(batch_size, device=device) + 0.8
        latent_noise = torch.randn(batch_size, NOISE_DIM, 1, 1, device=device)

        # add generator graph to tensorboard
        if epoch == 0 and idx == 0:
            writer.add_graph(generator, input_to_model=latent_noise)

        with torch.autocast(device_type=device, dtype=torch.float32, enabled=use_amp):
            # Train discriminator on batch of real data ------------------------------
            real_outputs = discriminator(noisy_real_data).reshape(-1)

            # Train discriminator on batch of fake data ------------------------------
            fake_data = generator(latent_noise)
            noisy_fake_data = gaussian_noise(fake_data.detach(), 0, 0.05)
            fake_outputs = discriminator(noisy_fake_data).reshape(-1)

        disc_real_loss = loss_fn(real_outputs, real_labels)
        disc_fake_loss = loss_fn(fake_outputs, fake_labels)
        disc_real_loss.backward()
        disc_fake_loss.backward()
        disc_optimizer.step()

        disc_loss_sum = disc_real_loss + disc_fake_loss

        # Train generator -------------------------------------------------------
        real_labels = torch.ones(batch_size, dtype=torch.float, device=device)
        with torch.autocast(device_type=device, dtype=torch.float32, enabled=use_amp):
            fake_outputs = discriminator(fake_data).reshape(-1)

        gen_loss = loss_fn(fake_outputs, real_labels)
        gen_scaler.scale(gen_loss).backward()
        gen_scaler.step(gen_optimizer)
        gen_scaler.update()

        # Show progress in terminal
        if idx % 50 == 0:
            print(f'[{epoch}/{EPOCHS}]'
                  f'[{idx}/{len(train_loader)}]\t'
                  f'Disc_Total_Loss: {disc_loss_sum.item():.4f}\t'
                  f'Gen_Loss: {gen_loss:.4f}'
                  )

        # Add batch losses to record
        batch_gen_losses = np.append(batch_gen_losses, gen_loss.item())
        batch_disc_losses = np.append(batch_disc_losses, disc_loss_sum.item())

    # END OF EPOCH ---------------------------------------------------------------

    # Update Records
    epoch_gen_loss = batch_gen_losses.mean().item()
    epoch_disc_loss = batch_disc_losses.mean().item()
    print(f'-----EPOCH {epoch} LOSS DISC: {epoch_disc_loss}, GEN: {epoch_gen_loss}-----')
    disc_scheduler.step()

    # Add to Tensorboard
    writer.add_scalar(tag="Learning Rate", scalar_value=torch.tensor(disc_scheduler.get_last_lr()), global_step=epoch)
    writer.add_scalars(main_tag="Loss", global_step=epoch, tag_scalar_dict={'Generator': epoch_gen_loss,
                                                                            'Disciminator': epoch_disc_loss})
    # Add Generator output images to tensorboard
    corrected_imgs = denormalize(fake_data[0:25])
    generator_grid = torchvision.utils.make_grid(corrected_imgs, 5, 5)
    writer.add_image(tag="gen_images", img_tensor=generator_grid, global_step=epoch)

# END OF TRAINING ----------------------------------------------------------------
print("Training Complete.\n")
writer.close()

# Option to overwrite saved best model
save = input("Save model?\n(y/n): ")
if save == "y":
    torch.save(generator, model_save_path)
    torch.save(generator.state_dict(), dict_save_path)
    print("Model Saved.")
