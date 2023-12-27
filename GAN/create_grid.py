import torchvision
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from data.cat_dataclass import CatDataset


transform = v2.Compose([
    v2.ToTensor(),
])

# Take the images placed in the gen_imgs_best folder and use torchvision to create an image grid
cat_faces = CatDataset(path=f'./imgs/gen_imgs_best', transforms=transform)
img_loader = DataLoader(cat_faces, batch_size=9, shuffle=False)

img_tensor = torch.empty((9, 3, 64, 64))
for idx, batch in enumerate(img_loader, 0):
    img_tensor = batch

grid = torchvision.utils.make_grid(img_tensor, nrow=3, padding=4)
torchvision.utils.save_image(grid, f"./imgs/cat_imgs/cat_gen_grid.png")
