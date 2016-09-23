import math
import numpy as np
import matplotlib.pyplot as plt 
import loadFittingDataP2 as fitting

def max_likelihood_weights_poly(X,Y,M):
	X_powers = np.array([[x**i for i in range(M+1)] for x in X])
	X_trans = np.transpose(X_powers)
	# theta = (X^T X)^(-1)X^T Y
	return np.dot(np.dot(np.linalg.inv(np.dot(X_trans,X_powers)),X_trans),Y)

def max_likelihood_weights_cos(X,Y,M):
	X_cos = np.array([[np.cos(math.pi*x*i) for i in range(1,M+1)] for x in X])
	X_trans = np.transpose(X_cos)
	# theta = (X^T X)^(-1)X^T Y
	return np.dot(np.dot(np.linalg.inv(np.dot(X_trans,X_cos)),X_trans),Y)

def squared_error(X,Y,M,theta):
	# least squares error: (X*theta-Y)^T (X*theta-Y)
	# gradient: X^T (X*theta-Y)
	X_powers = np.array([[x**i for i in range(M+1)] for x in X])
	diff = np.dot(X_powers,theta) - Y
	return np.dot(np.transpose(diff),diff), np.dot(np.transpose(X_powers),diff)

def ridge_regress(X,Y,M,lamb):
	X_powers = np.array([[x**i for i in range(M+1)] for x in X])
	X_trans = np.transpose(X_powers)
	prod = np.dot(X_trans,X_powers)+lamb*np.identity(len(np.dot(X_trans,X_powers)))
	# theta = (X^T X+lambda*I)^(-1)X^T Y
	return np.dot(np.dot(np.linalg.inv(prod),X_trans),Y)

def plot(X,y,n):
	theta = max_likelihood_weights_poly(X,y,n)
	x_n = np.linspace(0,1,100)
	y_n = [np.dot(theta,[x_n[i]**j for j in range(n+1)]) for i in range(100)]
	plt.plot(x_n, y_n)
	plt.show()

X,y = fitting.getData()
X = np.array(X)

plot(X,y,0)
plot(X,y,1)
plot(X,y,3)
plot(X,y,10)

theta = max_likelihood_weights_poly(X,y,1)
print squared_error(X,y,1,theta)

print max_likelihood_weights_cos(X,y,1)
print max_likelihood_weights_cos(X,y,2)
print max_likelihood_weights_cos(X,y,3)
print max_likelihood_weights_cos(X,y,4)
print max_likelihood_weights_cos(X,y,5)
print max_likelihood_weights_cos(X,y,6)
print max_likelihood_weights_cos(X,y,7)
print max_likelihood_weights_cos(X,y,8)

print max_likelihood_weights_poly(np.array(X),np.array(y),1)
print ridge_regress(np.array(X),np.array(y),1,0)
print ridge_regress(np.array(X),np.array(y),1,1)
