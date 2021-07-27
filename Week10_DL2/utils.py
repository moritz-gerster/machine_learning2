import matplotlib
from matplotlib import pyplot as plt

import torchvision

def getdata():

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

    Xr = trainset.data.float().view(-1,784)/127.5-1
    Xt = testset.data.float().view(-1,784)/127.5-1

    return Xr,Xt

def vis10(x):
    x = x.data.numpy().reshape(1,10,28,28).transpose(0,2,1,3).reshape(28,280)
    plt.figure(figsize=(8,1))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(x,cmap='gray')
    plt.show()

