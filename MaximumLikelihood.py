import math
import numpy as np
import matplotlib.pyplot as plt 
import loadFittingDataP2 as fitting
from GradientDescent import batch_gradient_descent
from GradientDescent import stochastic_gradient_descent
from GradientDescent import learning_rate
from GradientDescent import gradient_approx
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

def squared_error(theta, X,Y,M=3):
	# least squares error: (X*theta-Y)^T (X*theta-Y)
	X_powers = np.array([[x**i for i in range(M+1)] for x in X])
	diff = np.dot(X_powers,theta) - Y
	return np.dot(np.transpose(diff),diff)

def squared_error_deriv(theta,X,Y,M=3):
	# print "\ntheta: " + str(theta) + "\nx: " + str(X) + "\ny: " + str(Y)
	# gradient: X^T (X*theta-Y)
	X_powers = np.array([[x**i for i in range(M+1)] for x in X])
	diff = np.dot(X_powers,theta) - Y
	return 2*np.dot(np.transpose(X_powers),diff)



def ridge_regress(X,Y,M,lamb):
	X_powers = np.array([[x**i for i in range(M+1)] for x in X])
	X_trans = np.transpose(X_powers)
	prod = np.dot(X_trans,X_powers)+lamb*np.identity(len(np.dot(X_trans,X_powers)))
	# theta = (X^T X+lambda*I)^(-1)X^T Y
	return np.dot(np.dot(np.linalg.inv(prod),X_trans),Y)

def plot_poly(X,y,n):
	theta = max_likelihood_weights_poly(X,y,n)
	theta_r = ridge_regress(np.array(X),np.array(y),n,10)
	x_n = np.linspace(0,1,100)
	y_n = [np.dot(theta,[x_n[i]**j for j in range(n+1)]) for i in range(100)]
	yr_n = [np.dot(theta_r,[x_n[i]**j for j in range(n+1)]) for i in range(100)]
	f_n = [math.cos(3.1415*i/100) + math.cos(2*3.1415*i/100) for i in range(100)]
	plt.plot(x_n, y_n, X, y, 'o', x_n, f_n, x_n, yr_n)
	plt.show()

def plot_cos(X,y,n):
	theta = max_likelihood_weights_cos(X,y,n)
	x_n = np.linspace(0,1,100)
	y_n = [np.dot(theta,[np.cos(math.pi*x_n[i]*j) for j in range(1,n+1)]) for i in range(100)]
	f_n = [math.cos(3.1415*i/100) + math.cos(2*3.1415*i/100) for i in range(100)]
	plt.plot(x_n, y_n, X, y, 'o', x_n, f_n)
	plt.show()

X,y = fitting.getData()
X = np.array(X)
print X
print y
'''
print max_likelihood_weights_cos(X,y,1)
print max_likelihood_weights_cos(X,y,2)
print max_likelihood_weights_cos(X,y,3)
print max_likelihood_weights_cos(X,y,4)
print max_likelihood_weights_cos(X,y,5)
print max_likelihood_weights_cos(X,y,6)
print max_likelihood_weights_cos(X,y,7)
print max_likelihood_weights_cos(X,y,8)


theta=[6,1,9,4]
print squared_error(theta, X, y)
grad = squared_error_deriv(theta, X, y)
grad_app = gradient_approx(X, y, squared_error, theta, 2)
print grad
print grad_app
print np.linalg.norm(grad-grad_app)
batch = batch_gradient_descent(X, y, squared_error, squared_error_deriv,[i for i in range(5)], 10**-1, 10**-1) 
print batch

stoch =  stochastic_gradient_descent(X, y, squared_error, squared_error_deriv,[i for i in range(5)], learning_rate, 10**-1) 
print stoch

xb = [i for i in range(len(batch[2]))]
xs = [i for i in range(len(stoch[2]))]

#plt.plot(xb,batch[2],xs,stoch[2])
#plt.show()

#print max_likelihood_weights_poly(X,y,4)

plot_cos(X,y,0)
plot_cos(X,y,1)
plot_cos(X,y,3)
plot_cos(X,y,10)
theta = max_likelihood_weights_poly(X,y,1)
#print squared_error(X,y,1,theta)
plot_poly(X,y,1)
plot_poly(X,y,4)
plot_poly(X,y,8)
print max_likelihood_weights_poly(np.array(X),np.array(y),1)
print ridge_regress(np.array(X),np.array(y),1,0)
print ridge_regress(np.array(X),np.array(y),1,1)
print ridge_regress(np.array(X),np.array(y),1,5)
print ridge_regress(np.array(X),np.array(y),1,10)

'''
