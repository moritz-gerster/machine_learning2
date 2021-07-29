import torch
import torch.optim as optim
import torch.nn as nn
import numpy,numpy.random

class NNClassifier:

    def __init__(self,net,flat=False):
        self.net = net
        self.flat = flat

    def fit(self,X,T,mb=100,lr=0.01,epochs=1):

        if self.flat: X = X.view(len(X),-1)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)

        for epoch in range(epochs):
            for i in range(len(X)//mb):

                x = X[mb*i:mb*(i+1)]
                t = T[mb*i:mb*(i+1)]
                
                optimizer.zero_grad()
                criterion(self.net.forward(x).view(mb,-1),t).backward()
                optimizer.step()
            
        return self

    def predict(self,X):

        if self.flat: X = X.view(len(X),-1)

        return self.net.forward(X).view(len(X),-1).data

def graphdata(N=3000,m=15):

    numpy.random.seed(0)

    A = numpy.zeros([N,m,m])
    T = numpy.zeros([N])
    
    # Stars
    for i in range(N//3):
        j = numpy.random.randint(0,m)
        A[i,j,:] = 1.0; A[i,:,j] = 1.0; A[i,j,j] = 0.0; T[i] = 0

    # Chains
    for i in range(N//3,(N*2)//3):
        p = numpy.random.permutation(m)
        A[i,p[:-1],p[1:]] = 1.0; A[i,p[1:],p[:-1]] = 1.0; T[i] = 1

    # Random
    for i in range((N*2)//3,N):
        r = numpy.arange(m)
        p = numpy.random.permutation(m)
        A[i,r,p] = 1.0; A[i,p,r] = 1.0; T[i] = 2

    # Partitioning
    P = numpy.random.permutation(N)
    Ptr,Ptt = P[:N//2],P[N//2:]
    Ar = torch.FloatTensor(A[Ptr])
    Tr = torch.LongTensor(T[Ptr])
    At = torch.FloatTensor(A[Ptt])
    Tt = torch.LongTensor(T[Ptt])

    return Ar,Tr,At,Tt

