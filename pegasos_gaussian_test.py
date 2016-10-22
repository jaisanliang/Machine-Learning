import numpy as np
from numpy import *
from plotBoundary import *
import pylab as pl
# import your LR training code

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
epochs = 1000;
l = .02;
gamma = 2e-2;
n = X.shape[0]

K = np.zeros((n,n));
for i in range(n):
	for j in range(n):
		K[i][j] = np.exp(-gamma*np.linalg.norm(X[i]-X[j])**2)


def pegasos_kernel_train(X,Y,l,K,max_epochs):
	n = X.shape[0]
	d = X.shape[1]
	t = 0
	alpha = np.zeros(n)
	for epoch in range(max_epochs):
		for i in range(n):
			t += 1
			nu_t = 1.0/(t*l)
			if Y[i][0]*np.dot(alpha,K[i,:]) < 1:
				alpha[i] = (1-nu_t*l)*alpha[i] + nu_t*Y[i][0]
			else:
				alpha[i] = (1-nu_t*l)*alpha[i]
	return alpha

alpha = pegasos_kernel_train(X,Y,l,K,epochs)

# Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
def predict_gaussianSVM(x):
	n = X.shape[0]
	K_i = np.zeros(n)
	for i in range(n):
		K_i[i] = np.exp(-gamma*np.linalg.norm(X[i]-x)**2)
	return np.dot(alpha,K_i)

# plot training results
plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM')
pl.show()
