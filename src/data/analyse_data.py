from src.data.utils import get_nodes_edges
import numpy as np
import os
import matplotlib
from matplotlib import collections  as mc
import pylab as pl
import matplotlib.pyplot as plt

def get_city_centroids(datadir):

    centroids = {}
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

            centroid_x = np.sum(xlim) / 2
            centroid_y = np.sum(ylim) / 2

            centroids[name] = (centroid_x, centroid_y)
        except:
            print(fn)

    return centroids

