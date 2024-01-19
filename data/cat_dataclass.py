import os
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset


class CatDataset(Dataset):
    def __init__(self, path, transforms):
        if os.path.exists(path):
            self.dataset_dir = Path(path)
            self.all_filenames = os.listdir(self.dataset_dir)
        else:
            self.dataset_dir = None
            self.all_filenames = []

        self.transforms = transforms

    def __len__(self):
        return len(self.all_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, self.all_filenames[idx])
        image_pil = Image.open(image_path).convert('RGB')
        sample = self.transforms(image_pil)

        return sample

