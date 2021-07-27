import numpy,numpy.core

# -------------------------------------------
# Remove random nucleotides
# -------------------------------------------
def remove(X,rstate):
	Y = []
	for x in X:
		l = rstate.randint(8,12)
		Y += [x[l]]
		x[l] = '?'
	return X,Y

# -------------------------------------------
# Load gene sequence data
# -------------------------------------------
def loaddata():

	Xtrain = numpy.array([numpy.array(list(l[20:40])) for l in open('splice-data/splice-train-data.txt','r')])
	Xtest  = numpy.array([numpy.array(list(l[20:40])) for l in open('splice-data/splice-test-data.txt','r')])

	Xtrain,Ytrain = remove(Xtrain,numpy.random.mtrand.RandomState(1234))
	Xtest, Ytest  = remove(Xtest,numpy.random.mtrand.RandomState(2345))

	return Xtrain,Xtest,Ytrain,Ytest

# -------------------------------------------
# Compute degree kernels
# -------------------------------------------
def getdegreekernels(Xtrain,Xtest,degree):

	na = numpy.newaxis

	Ztrain = Xtrain
	Ztest  = Xtest

	for i in range(degree-1):
		Ztrain = numpy.core.defchararray.add(Ztrain[:,:-1],Xtrain[:,i+1:])
		Ztest  = numpy.core.defchararray.add(Ztest[:,:-1],Xtest[:,i+1:])

	Ktrain = (Ztrain[:,na,:] == Ztrain[na,:,:]).sum(axis=2)
	Ktest  = (Ztest[:,na,:]  == Ztrain[na,:,:]).sum(axis=2)

	return Ktrain,Ktest

