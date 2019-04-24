from src.visualisation.visualise import find_closest_cities

from src.data.dataset import CityImageDataset
from src.data.invert import Invert
from settings import FIGPATH, PROJECT_PATH
import os
import sys
import torch
from torchvision import transforms

# Prepare data loader
datapath = os.path.join(FIGPATH,'figures_20000')

train_dataset = CityImageDataset(root=os.path.join(datapath,'train'),transform=transforms.Compose([
    transforms.Grayscale(),
    Invert(),
    transforms.CenterCrop(128),    # instead of random crop
    transforms.Resize(64),
    transforms.ToTensor(),
]))
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1
)

# Set model path
PROJECT_PATH
model_path = os.path.join(PROJECT_PATH,'src','models','vae_random.torch')
print(model_path)
outputdir = os.path.join(PROJECT_PATH,'reports','figures_closest_random_2')

# Find closest cities
find_closest_cities(no_cities=100, model_path=model_path,output_dir=outputdir,data_loader=train_loader)
