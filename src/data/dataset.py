from torch.utils import data
from src.data.invert import Invert
from torchvision import datasets
# from skimage import io, transform
import os
import torch
import numpy as np

from PIL import Image

class CityImageDataset(data.Dataset):
    """
    Urban images dataset.
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [f for f in os.listdir(root) if not f.startswith('.')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root,self.image_files[idx])
        sample = Image.open(img_name)

        if self.transform:
            sample = self.transform(sample)

        return sample, 0, img_name


class Normalize(object):
    """Remove mean image from tensor."""

    def __init__(self, mean_image):
        assert isinstance(mean_image, np.ndarray)
        assert (len(mean_image.shape) == 2)

        mean_image_expanded = np.expand_dims(mean_image, axis=0)
        self.mean_image = torch.from_numpy(mean_image_expanded)

    def __call__(self, sample):
        image_normalized = sample - self.mean_image
        return image_normalized






