import math
import numpy as numpy

def max_likelihood_weights_poly(X,Y,M):
	X_powers = np.array([[x**i for i in range(M+1)] for x in X])
	X_trans = np.transpose(X_powers)
	# theta = (X^T X)^(-1)X^T Y
	return np.dot(np.dot(np.linalg.inverse(np.dot(X_trans,X_powers)),X_trans),Y)

def max_likelihood_weights_cos(X,Y,M):
	X_powers = np.array([[np.cos(math.pi*x*i) for i in range(M+1)] for x in X])
	X_trans = np.transpose(X_powers)
	# theta = (X^T X)^(-1)X^T Y
	return np.dot(np.dot(np.linalg.inverse(np.dot(X_trans,X_powers)),X_trans),Y)

def squared_error(X,Y,M,theta):
	# least squares error: (X*theta-Y)^T (X*theta-Y)
	# gradient: X^T (X*theta-Y)
	X_powers = np.array([[x**i for i in range(M+1)] for x in X])
	diff = np.dot(X_powers,theta) - Y
	return np.dot(np.transpose(diff),diff), np.dot(np.transpose(X_powers),diff)

def ridge_regress(X,Y,lamb):
	# get dimension of X vectors
	prod = np.dot(np.transpose(X),X)+lamb*np.identity(len(X)[0])
	# theta = (X^T X+lambda*I)^(-1)X^T Y
	return np.dot(np.dot(np.linalg.inverse(prod),X_trans),Y)