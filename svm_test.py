from numpy import *
import numpy as np
from plotBoundary import *
import pylab as pl
from cvxopt import matrix, solvers

# parameters
name = '3'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()
n = X.shape[0]
C = 10000

# Carry out training, primal and/or dual
Y_mat = np.dot(Y,np.transpose(Y))
X_mat = np.dot(X,np.transpose(X))
P = matrix(np.multiply(Y_mat,X_mat)) #P = Y Y^T * X X^T where * is element wise multiplication
q = matrix(np.array([-1 for i in range(n)]).astype(np.double)) # q = -[1 1 ... 1]^T where there are n ones
G = matrix(np.transpose(np.hstack((np.identity(n),-1*np.identity(n)))).astype(np.double)) # G = [I I]^T where the Is are n by n and stacked on top of each other
h = matrix(np.array([C for i in range(n)] + [0 for i in range(n)]).astype(np.double)) # h = [C 0]^T where there are n Cs and n 0s
A = matrix(np.transpose(Y).astype(np.double)) # A = Y^T
b = matrix(np.array([[0]]).astype(np.double)) # b = 0
# find the solution	
solution = solvers.qp(P, q, G, h, A, b)
xvals = np.array(solution['x'])
#print xvals
w = np.zeros(X.shape[1])
for i in range(len(xvals)):
	w = np.add(w, xvals[i][0]*Y[i][0]*X[i])

b = 0
epsilon = 1e-3
#print xvals
for i in range(len(xvals)):
	if xvals[i][0] > (0 + epsilon) and xvals[i][0] < (C - epsilon):
		b = Y[i][0]-np.dot(w,X[i])
		break
#print b

def predictSVM(x):
	return np.dot(w,x) + b

# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')
pl.show()

'''
print '======Validation======'
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]
# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
pl.show()
'''

