import os
from natsort import natsorted
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CatDataset(Dataset):
    # Cat Dataset for 64 x 64 images for color classification.
    def __init__(self):
        self.dataset_dir = os.path.join(os.getcwd(), 'dataset')
        self.all_filenames = natsorted(os.listdir(self.dataset_dir))

    def __len__(self):
        return len(self.all_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, self.all_filenames[idx])
        image_pil = Image.open(image_path).convert('RGB')
        image_tensor = transforms.ToTensor()(image_pil)

        sample = {'image': image_tensor, 'idx': idx}

        return sample
