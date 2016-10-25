from numpy import *
import numpy as np
from plotBoundary import *
import pylab as pl
from cvxopt import matrix, solvers



def k_linear(x,z):
	return np.dot(x,z)

RBF_BANDWIDTH = 1
def k_rbf(x,z):
	return math.exp(-1/(2*RBF_BANDWIDTH**2) * np.linalg.norm(x-z)**2) 

k_sel = { # stores kernel functions
	'linear': k_linear,
	'rbf': k_rbf
}
k = 'linear'

# minimize 1/2 |sum(alpha_i y^(i) x^(i))|^2 + sum(alpha_i)
# s.t. sum(alpha_i y^(i)) = 0
# where 0 <= alpha_i <= C
# 
# in form min 1/2 x^T P x + q^T x
# s.t. Ax = b
# where Gx <= h
	
def svm(dataset, k, C):
	print '========== Training =========='
	print 'Dataset: '+str(dataset)+ ' ||| C: '+str(C)	
	train = loadtxt('data/data'+str(dataset)+'_train.csv')
	X = train[:, 0:2].copy()
	Y = train[:, 2:3].copy()
	
	n = X.shape[0]

	# convert data into solver form
	Y_mat = np.dot(Y,np.transpose(Y))
	K = np.array([[k_sel[k](x,z) for x in X] for z in X])
	P = matrix(np.multiply(Y_mat,K)) #P = Y Y^T * K where * is element wise multiplication
	q = matrix(np.array([-1 for i in range(n)]).astype(np.double)) # q = -[1 1 ... 1]^T where there are n ones

	A = matrix(np.transpose(Y).astype(np.double)) # A = Y^T
	b = matrix(np.array([[0]]).astype(np.double)) # b = 0
	
	G = matrix(np.transpose(np.hstack((np.identity(n),-1*np.identity(n)))).astype(np.double)) # G = [I -I]^T where the Is are n by n and stacked on top of each other
	h = matrix(np.array([C for i in range(n)] + [0 for i in range(n)]).astype(np.double)) # h = [C 0]^T where there are n Cs and n 0s
	
	# run through quadratic solver	
	solution = solvers.qp(P, q, G, h, A, b)
	xvals = np.array(solution['x']) # alpha_i's

	# calculate b
	b = 0
	epsilon = 1e-3
	for i in range(len(xvals)):
		if xvals[i][0] > (0 + epsilon) and xvals[i][0] < (C - epsilon):
			#b = Y[i][0] - np.dot(w,X[i])
			kv = np.array([[k_sel[k](X[i],j)] for j in X])
			b = Y[i][0] - np.dot(np.transpose(xvals),np.multiply(Y,kv))
	
	# print data
	mc = 0 # misclassified	
	cc = 0 # correctly classified
	for i in range(len(xvals)):
		if xvals[i][0] < (0 + epsilon):
			cc = cc + 1
		elif xvals[i][0] > (C - epsilon):
			mc = mc + 1 
	
	print 'Correct | Incorrect | Support Vectors'
	print str(cc) + ' | ' + str(mc) + ' | ' + str(n-cc-mc)
	print str(cc/float(n)) + ' | ' + str(mc/float(n)) + ' | ' + str((n-cc-mc)/float(n))

		
	def predictSVM(x):
		kv = np.array([[k_sel[k](x,j)] for j in X])
		return np.dot(np.transpose(xvals),np.multiply(Y,kv)) + b
	
	result = {
		'X': X,
		'Y': Y,
		'C': C,
		'predict': predictSVM,
		'alphas': xvals,
	}
	return result
	

def plot(result, title_in):
	plotDecisionBoundary(result['X'], result['Y'], result['predict'], [-1, 0, 1], title = title_in)
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
def find_params(result):	
	X = result['X']
	Y = result['Y']
	C = result['C']
	xvals = result['alphas']
	epsilon = 1e-3
	w = np.zeros(X.shape[1])
	for i in range(len(xvals)):
		w = np.add(w, xvals[i][0]*Y[i][0]*X[i])	
	for i in range(len(xvals)):
		if xvals[i][0] > (0 + epsilon) and xvals[i][0] < (C - epsilon):
			b = Y[i][0] - np.dot(w,X[i])
	return {'w': w, 'b': b}

# svm(dataset number, kernel type, c)
# 2.1
result = svm(5, 'linear', 1)
params = find_params(result)
#plot(result,'Trivial SVM')
print params
'''
for i in [.01,.1,1,10,100]:
	result = svm(1, 'linear', i)
	plot(result['X'], result['Y'], result['predict'], 'Trivial SVM. C = ' + str(i))
'''
