import numpy
import copy
import matplotlib
import torch
import torch.nn
from matplotlib import pyplot as plt

def newlayer(layer, g):
    layer = copy.deepcopy(layer)

    try:
        layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError:
        pass

    try:
        layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError:
        pass

    return layer


def visualize(R):
    R = R[0].sum(dim=0).data.numpy()

    b = 10*((numpy.abs(R)**3.0).mean()**(1.0/3))

    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)

    plt.figure(figsize=(3,3))
    plt.axis('off')
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.imshow(R,cmap=my_cmap,vmin=-b,vmax=b)

