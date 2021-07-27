import matplotlib
from matplotlib import pyplot as plt

import torchvision

def getdata():

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

    Tr = trainset.targets.data.numpy()
    Tt = testset.targets.data.numpy()

    Xr = (trainset.data.float().view(-1,784)/255.0).data.numpy()
    Xt = (testset.data.float().view(-1,784)/255.0).data.numpy()

    return Xr[Tr==0][:100]*1.0,Xt[Tt==0]*1.0,Xt[Tt!=0]*1.0

def vis10(x):
    x = x.reshape(1,10,28,28).transpose(0,2,1,3).reshape(28,280)
    plt.figure(figsize=(8,1))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(x,cmap='gray')
    plt.show()

