import numpy
import matplotlib
from matplotlib import pyplot as plt

M1,M2 = numpy.meshgrid(numpy.linspace(-12,12,100),numpy.linspace(-12,12,100))
Xgrid = numpy.array([M1.flatten(),M2.flatten()]).T

def plot(X,F,boundary=False):

    plt.figure(figsize=(4,4))
    if F is not None: plt.contourf(M1,M2,F.reshape(M1.shape),cmap='Reds')
    if F is not None and boundary: plt.contour(M1,M2,F.reshape(M1.shape),colors='black',levels=[0])
    plt.scatter(*X.T,color='black',s=5)
    plt.xlim(-12,12)
    plt.ylim(-12,12)
    plt.show()

