import numpy as np
import pylab as pl
from matplotlib import collections  as mc
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib
import torch

from settings import FIGPATH
from src.data.utils import get_nodes_edges
from src.models.vae import VAE
import random
from torchvision.utils import save_image
from sklearn.metrics import pairwise_distances
import os
from settings import PROJECT_PATH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_cities(datadir, window_size=None, size_inches = (10.5,10.5)):
    """

    :param datadir:
    :param window_size: window size in meters
    :param size_inches:
    :return:
    """

    # Create fig directory
    if window_size is not None:
        figpath = os.path.join(FIGPATH,'figures_'+str(window_size))
    else:
        figpath = FIGPATH
    if not os.path.exists(figpath):
        os.makedirs(figpath)

    # Additionally, collect information on image ranges (remove later)
    xranges = []
    yranges = []

    filenames_coords, filenames_edges = get_nodes_edges(datadir)
    print(len(filenames_edges))

    for fn in filenames_coords:
        try:
            name = fn.rstrip('_COORDS.txt')
            nodes = np.loadtxt(os.path.join(datadir,fn),skiprows=1)
            edges = np.loadtxt(os.path.join(datadir,name+'_NCOL.txt'),skiprows=1)
            nodes_dict = dict(zip(nodes[:,0],nodes[:,1:]))
            lines = [[nodes_dict[edge[0]],nodes_dict[edge[1]]] for edge in edges]
            weights = [0.1*np.log(edge[2]) for edge in edges]

            lc = mc.LineCollection(lines, linewidths=weights, colors='k')
            fig, ax = pl.subplots()
            ax.add_collection(lc)
            ax.autoscale()
            ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            plt.axis("off")

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            if window_size is not None:
                centroid_x = np.sum(xlim) / 2
                centroid_y = np.sum(ylim) / 2

                radius = window_size/2
                plt.xlim(centroid_x - radius, centroid_x + radius)
                plt.ylim(centroid_y - radius, centroid_y + radius)

            fig.set_size_inches(size_inches[0], size_inches[1])

            plt.savefig('%s/%s.png' % (figpath, name),dpi=400,bbox_inches='tight', pad_inches = 0)
            plt.close()

            xranges.append(xlim[1]-xlim[0])
            yranges.append(ylim[1]-ylim[0])
        except Exception as e:
            print("Filename:",fn)
            print(e)

    return xranges, yranges


def find_closest_cities(no_cities, model_path,data_loader,output_dir, no_closest=10):

    # outputdir = os.path.join(PROJECT_PATH,'reports','figures_closest')

    # Load model
    print("Load model")
    image_channels = 1
    model = VAE(image_channels=image_channels, h_dim=1024).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))  # Start with pretrained model.

    # Get data
    print("Get data")
    image_names = []
    image_tensors = []
    z_images = np.zeros((848, 32))
    with torch.no_grad():
        for idx, (image, _, image_name) in enumerate(data_loader):
            z_image = model.encode(image)[0].numpy()
            z_images[idx] = z_image
            image_names.append(image_name)
            image_tensors.append(image)

    # Find closest
    print("Find closest")
    for i in range(0, no_cities):

        # Get random index
        idx = random.choice(range(0, len(z_images)))

        # Create output folder
        folder_name1 = model_path.split('/')[-1]
        folder_name2 = image_names[idx][0].split('/')[-1].rstrip('.png')
        folder_path = os.path.join(output_dir, folder_name1,folder_name2)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save orig image
        img_name = '0_' + image_names[idx][0].split('/')[-1]
        save_image(image_tensors[idx].cpu(), os.path.join(folder_path, img_name))

        # Get distances
        dists = pairwise_distances(np.array(z_images), z_images[idx].reshape(1, -1))
        dists[idx] = 1000
        min_dists = sorted(zip(range(0, len(dists)), dists), key=lambda t: t[1])[0:no_closest]

        for i, (idx_min, dist) in enumerate(min_dists):
            img_name_closest = str(i + 1) + '_' + image_names[idx_min][0].split('/')[-1]
            save_image(image_tensors[idx_min].cpu(), os.path.join(folder_path, img_name_closest))
