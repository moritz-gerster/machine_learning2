import numpy
import matplotlib
import matplotlib.pyplot as plt

def showimage(i):
    i = i.astype('float')
    i = i-i.min()
    i = i/i.max()
    plt.figure()
    plt.axis('off')
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.imshow(i)
    plt.show()

def showpatches(p):
    p = p.astype('float')
    p = p-p.min()
    p = p/p.max()
    plt.figure(figsize=(8,2))
    plt.axis('off')
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.imshow(p[:64].reshape(4,16,8,8,3).transpose(0,2,1,3,4).reshape(4*8,16*8,3))
    plt.show()

