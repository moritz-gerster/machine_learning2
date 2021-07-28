import numpy

na = numpy.newaxis

# Hidden Markov Model class
class HMM:
    
    # create a randomly initialized HMM
    def __init__(self,N,M):
        
        # create random initial state
        self.Pi = numpy.ones([N])/N
        
        # initialize HMM parameters to random positive values
        self.A = numpy.random.exponential(1,[N,N])
        self.B = numpy.random.exponential(1,[N,M])
        
        # normalize probability distributions to 1
        self.A = self.A / self.A.sum(axis=1)[:,na]
        self.B = self.B / self.B.sum(axis=1)[:,na]

    # Load a sequence of observations
    def loaddata(self,O):
        
        self.O = O
        self.Z = numpy.array([self.B[:,self.O[t]] for t in range(len(self.O))])
    
    # Forward procedure (Eq. 19-21)
    def forward(self):
        
        # Initialization
        self.alpha = numpy.empty([len(self.O),len(self.Pi)])
        self.alpha[:] = numpy.NaN
        self.alpha[0] = self.Pi*self.Z[0]
        
        # Induction
        for t in range(len(self.O)-1):
            self.alpha[t+1] = numpy.dot(self.alpha[t],self.A)*self.Z[t+1]
        
        # Termination
        self.pobs = self.alpha[-1].sum()

    # Backward procedure (Eq. 24-25)
    def backward(self):
        
        # Initialization
        self.beta = numpy.empty([len(self.O),len(self.Pi)])
        self.beta[:] = numpy.NaN
        self.beta[-1] = 1.0
        
        # Induction
        for t in range(len(self.O)-2,-1,-1):
            self.beta[t] = numpy.dot(self.beta[t+1]*self.Z[t+1],self.A.T)
    
    # Baum-Welch parameter update (Eq. 36-40)
    def learn(self):
        
        # Compute gamma
        self.gamma = self.alpha*self.beta / self.pobs
        
        # Compute xi and psi
        self.xi = self.alpha[:-1,:,na]*self.A[na,:,:]*self.beta[1:,na,:]*self.Z[1:,na,:] / self.pobs
        self.psi = self.gamma[:,:,na]*(self.O[:,na,na] == numpy.arange(self.B.shape[1])[na,na,:])
        
        # Update HMM parameters
        self.A  = (self.xi.sum(axis=0)  / self.gamma[:-1].sum(axis=0)[:,na])
        self.B  = (self.psi.sum(axis=0) / self.gamma.sum(axis=0)[:,na])
        self.Pi = (self.gamma[0])

