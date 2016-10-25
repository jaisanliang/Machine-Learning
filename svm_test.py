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

def load_data(dataset, datatype):
	print 'Loading dataset: ' + str(datatype) + ' ' + str(dataset) 
	train = loadtxt('data/data' + str(dataset) + '_' + datatype + '.csv')
	X = train[:, 0:2].copy()
	Y = train[:, 2:3].copy()
	return [X, Y]

# minimize 1/2 |sum(alpha_i y^(i) x^(i))|^2 + sum(alpha_i)
# s.t. sum(alpha_i y^(i)) = 0
# where 0 <= alpha_i <= C
# 
# in form min 1/2 x^T P x + q^T x
# s.t. Ax = b
# where Gx <= h
	
def svm(X, Y, k, C):
	print '========== Training =========='
	
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
		'predictSVM': predictSVM,
		'alphas': xvals,
	}
	return result

# Given alphas, find the appropriate w, b (for w^T x + b)
# Assume linear kernel (aka no kernel)
def find_params(X, Y, result):
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

def validate(X, Y, result):	
	print '======Validation======'
	correct = 0;
	n = X.shape[0]
	for i in range(n):
		if ( Y[i] == (1 if result['predictSVM'](X[i]) > 0 else -1) ) : # we want sign(predictSVM(x)) 
			correct = correct + 1
	print 'Correct | Incorrect'
	print str(correct) + ' | ' + str(n-correct)
	print str(correct/float(n)) + ' | ' + str((n-correct)/float(n))
	return (n-correct)/float(n)
	
# plot a given function and data
def plot(X, Y, predictSVM, title_in):
	plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = title_in)
	pl.show()

# svm(X, Y, kernel type, c)
'''
# 2.1
# Show constraints and objective for 4 point example (dataset 5)
# Which examples are support vectors 
X, Y = load_data(5, 'train')
result = svm(X, Y, 'linear', 1)
params = find_params(X, Y, result)
plot(X, Y, result['predictSVM'],'Trivial SVM')
print params
'''
'''	
# 2.2
# Set C = 1, report / explain decision boundary and classification error rate on training / validation
for i in range(1,5):
	X, Y = load_data(i, 'train')
	result = svm(X, Y, 'linear', 1)
	plot(X, Y, result['predictSVM'], 'Training Dataset ' + str(i))
	Xv, Yv = load_data(i, 'validate')
	validate(Xv, Yv, result)
	plot(Xv, Yv, result['predictSVM'], 'Validation Dataset ' + str(i))		
'''

# 2.3 
# include kernels
# Explore effects of C in .01, .1, 1, 10, 100 on linear kernels and gaussian rbfs (with varying bandwidth)
'''
pl.plot([-2, -1, 0, 1, 2], [0, 33, 76, 78, 74])
pl.xlabel('log(C)')
pl.title('No. of SV')
pl.rcParams.update({'font.size': 25})
pl.show()
'''
'''
for i in [.01,.1, 1, 10, 100]:
	X, Y = load_data(2, 'train')
	result = svm(X, Y, 'linear', i)
	X, Y = load_data(2, 'validate')
	plot(X, Y, result['predictSVM'],  'Linear SVM. C = ' + str(i))


for j in [.1, 1, 10]:
	RBF_BANDWIDTH = j
	for i in [.01, .1, 1, 10, 100]:
		X, Y = load_data(4, 'train')
		result = svm(X, Y, 'rbf', i)
		X, Y = load_data(4, 'validate')
		plot(X, Y, result['predictSVM'], 'RBF SVM. C = ' + str(i))
'''
'''
table = []
for i in [1,2,3,4]:
	for j in [.01, .1, 1, 10, 100]:
		X, Y = load_data(i, 'train')
		lin_result = svm(X, Y, 'linear', j)
		RBF_BANDWIDTH = .1
		s01_result = svm(X, Y, 'rbf', j)
		RBF_BANDWIDTH = 1
		s1_result = svm(X, Y, 'rbf', j)
		RBF_BANDWIDTH = 10
		s10_result = svm(X, Y, 'rbf', j)
		X, Y = load_data(i, 'validate') 
		data = [validate(X, Y, lin_result), validate(X, Y, s01_result), validate(X, Y, s1_result), validate(X, Y, s10_result)]
		table.append(data)

print table
'''
