import torch
import torchvision
from utils import denormalize

NUM_IMAGES = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)

cat_model = torch.load("saved_models/cat_model/best_gen.pth")
cat_model.to(device)
cat_model.eval()

# Generate a batch of images from noise using the save model
latent_noise = torch.randn(NUM_IMAGES, 100, 1, 1, device=device)
imgs = cat_model(latent_noise)
corrected_imgs = denormalize(imgs).to('cpu')

for i in range(len(corrected_imgs)):
    torchvision.utils.save_image(corrected_imgs[i], f"./imgs/gen_imgs/gen_{i}.png")
