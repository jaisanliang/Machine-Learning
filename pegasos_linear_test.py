import numpy as np
from numpy import *
from plotBoundary import *
import pylab as pl

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

def pegasos_train(X,Y,l,max_epochs):
	n = X.shape[0]
	d = X.shape[1]
	t = 0
	w = np.zeros(d)
	w_0 = 0
	for epoch in range(max_epochs):
		for i in range(n):
			t += 1
			nu_t = 1.0/(t*l)
			if Y[i][0]*(np.dot(w,X[i])+w_0) < 1:
				w = (1-nu_t*l)*w+nu_t*Y[i][0]*X[i]
				w_0 += Y[i][0]
			else:
				w = (1-nu_t*l)*w
	return w, w_0

def predict_linearSVM(x):
	return np.dot(w,x) + w_0

w, w_0 = pegasos_train(X,Y,2**(-10),1000)

# plot training results
plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
pl.show()

