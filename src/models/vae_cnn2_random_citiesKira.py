# Source: https://github.com/sksq96/pytorch-vae

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import os

from random import randint
from settings import FIGPATH, PROJECT_PATH
from src.data.dataset import CityImageDataset, Normalize
from src.data.load_data import train_test_split, get_images_mean
from src.data.invert import Invert

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
bs = 32
h_dim = 1024  # size of the output of CNN output layer ==> depends on image size (set to 1024 for 64x64 images)
log_interval = 5
datapath = os.path.join(PROJECT_PATH,'data','citiesKira','cropped')
modelpath = os.path.join(PROJECT_PATH,'models','citiesKira','random')
resultpath = os.path.join(PROJECT_PATH,'results','citiesKira','random')

# Logging setup
logging.basicConfig(filename=os.path.join(resultpath,'vae_cnn2_random.log'), filemode='w', format='%(message)s')


# Split data into train and test folders (only do once after data creation)
# train_test_split(datapath)

# Create train and test data loaders
# train_dataset_raw = CityImageDataset(
#     root=os.path.join(datapath,'train'),
#     transform = transforms.Compose([
#     transforms.Grayscale(),
#     Invert(),
#     transforms.RandomCrop(128),  # was 350
#     transforms.Resize(64),
#     transforms.ToTensor(),
# ])
# )
# image_mean = get_images_mean(train_dataset_raw)
print("Image mean calculated")
train_dataset = CityImageDataset(root=os.path.join(datapath,'train'),transform=transforms.Compose([
    transforms.Grayscale(),
    Invert(),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(), # added new transformation
    # transforms.CenterCrop(128),    # instead of random crop
    transforms.Resize(64),
    transforms.ToTensor(),
    # Normalize(image_mean),
]))
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=bs,
    shuffle=True
)

test_dataset = CityImageDataset(root=os.path.join(datapath,'test'),transform=transforms.Compose([
    transforms.Grayscale(),
    Invert(),
    transforms.CenterCrop(128),
    transforms.Resize(64),
    transforms.ToTensor(),
    # Normalize(image_mean),
]))

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=bs,
    shuffle=True
)

# Fixed input for debugging
fixed_x, _, _ = next(iter(test_loader))
print(fixed_x.size())
save_image(fixed_x, os.path.join(resultpath,'real_image.png'))


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=h_dim):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


image_channels = fixed_x.size(1)
model = VAE(image_channels=image_channels, h_dim=h_dim).to(device)
# model.load_state_dict(torch.load('vae_random.torch', map_location='cpu'))   # Start with pretrained model.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


def train(epoch):
    train_loss = 0
    train_bce = 0
    for idx, (images, _, _) in enumerate(train_loader):
        recon_images, mu, logvar = model(images)
        loss, bce, kld = loss_function(recon_images, images, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        train_bce += bce.item()
        optimizer.step()

        to_print = "Epoch[{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch,loss.data/bs,
                                                                    bce.data/bs, kld.data/bs)
        print(to_print)
        # if idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, idx * len(images), len(train_loader.dataset),
        #         100. * idx / len(train_loader),
        #         loss.data / len(images)))

    train_loss /= len(train_loader.dataset)
    train_bce /= len(train_loader.dataset)
    print('====> Train set loss: {:.4f} {:.4f}'.format(train_loss, train_bce))

    torch.save(model.state_dict(), os.path.join(modelpath,'vae_random.torch'))



def test(epoch):
    test_loss = 0
    test_bce = 0
    with torch.no_grad():
        for i, (data, _, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            test_bce += bce.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(bs, 1, 64, 64)[:n]])
                save_image(comparison.cpu(),
                         os.path.join(resultpath,'reconstruction_' + str(epoch) + '.png'), nrow=n)

    test_loss /= len(test_loader.dataset)
    test_bce /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f} {:.4f}'.format(test_loss, test_bce))
    # logging.warning("Epoch {} - test loss {:.4f} {:.4f}".format(epoch, test_loss, test_bce))  # uncomment if below code not running

    # The same of training data (remove if not interested in training performance so much)
    train_loss = 0
    train_bce = 0
    with torch.no_grad():
        for i, (data, _, _) in enumerate(train_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
            train_loss += loss.item()
            train_bce += bce.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(bs, 1, 64, 64)[:n]])
                save_image(comparison.cpu(),
                         os.path.join(resultpath,'reconstruction_train_' + str(epoch) + '.png'), nrow=n)

    train_loss /= len(train_loader.dataset)
    train_bce /= len(train_loader.dataset)
    print('====> Train set loss: {:.4f} {:.4f}'.format(train_loss, train_bce))
    logging.warning("Epoch {} - train loss {:.4f} {:.4f} test loss {:.4f} {:.4f}".format(epoch, train_loss, train_bce, test_loss, test_bce))


def compare(x):
    recon_x, _, _ = model(x)
    return torch.cat([x, recon_x])



if __name__ == "__main__":
    epochs = 500
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            print(torch.randn(32, 32).size())
            sample = torch.randn(32, 32).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(32, 1, 64, 64),
                       os.path.join(resultpath,'sample_' + str(epoch) + '.png'))

    fixed_x = test_dataset[randint(1, 100)][0].unsqueeze(0)
    compare_x = compare(fixed_x)

    save_image(compare_x.data.cpu(), os.path.join(resultpath,'sample_image.png'))