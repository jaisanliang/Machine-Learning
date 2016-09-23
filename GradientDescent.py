import math
import numpy as np
from numpy import *
import loadFittingDataP1 as fitting
import loadParametersP1 as parameters

'''
Performs gradient descent procedure to find minimum value min of obj_func
and parameters theta_min such that obj_fun(theta_min)=min
init_theta - d X 1 column vector with initial guess for theta_min
obj_func - scalar function of theta
grad_func - vector function of theta, d X 1 column vector
step_size - scalar that multiplies gradient when updating theta
threshold - if successive values of the objective function differ by less than threshold,
	stop the gradient descent procedure

Returns - theta_min, obj_func(theta_min)
'''
def gradient_descent(obj_func, grad_func, init_theta, step_size, threshold):
	theta_prev = init_theta
	theta = theta_prev - step_size * grad_func(theta_prev)
	count = 0
	while abs(obj_func(theta)-obj_func(theta_prev)) > threshold and count < 1000:
		print obj_func(theta)
		theta_prev = theta
		theta = theta_prev - step_size * grad_func(theta_prev)
		count += 1
	return theta, obj_func(theta)

'''
Central difference approximation for the gradient of a function
obj_func - scalar function of theta
theta - d X 1 column vector
delta - scalar controlling the difference step size

Returns - d X 1 gradient vector
'''
def gradient_approx(obj_func, theta, delta):
	return np.array([obj_func(theta+np.array([delta/2*(i==j) for j in range(len(theta))]))-obj_func(theta-np.array([delta/2*(i==j) for j in range(len(theta))])) for i in range(len(theta))])/delta

'''
Performs batch gradient descent procedure to find minimum value min of obj_func
and parameters theta_min such that obj_fun(theta_min)=min
data - n X d matrix containing n data vectors
init_theta - d X 1 column vector with initial guess for theta_min
obj_func - scalar function of data and theta
grad_func - vector function of data and theta, d X 1 column vector
step_size - scalar that multiplies gradient when updating theta
threshold - if successive values of the objective function differ by less than threshold,
	stop the gradient descent procedure

Returns - theta_min, obj_func(theta_min)
'''
def batch_gradient_descent(X, y, obj_func, grad_func, init_theta, step_size, threshold):
	theta_prev = init_theta
	theta = theta_prev - step_size * grad_func(theta_prev, X, y)
	while abs(obj_func(theta, X, y)-obj_func(theta_prev, X, y)) > threshold:
		theta_prev = theta
		theta = theta_prev - step_size * grad_func(theta_prev, X, y)
	return theta, obj_func(theta, X, y)

def mini_batch_gradient_descent(X, y, batch_size, obj_func, grad_func, init_theta, step_size, threshold):
	t_start = 0
	t_stop = batch_size
	theta_prev = init_theta
	batch_X = np.take(X, range(t_start, t_stop), mode = wrap)
	batch_y = np.take(y, range(t_start, t_stop), mode = wrap)
	theta = theta_prev - step_size * grad_func(theta_prev, batch_X, batch_y)
	while abs(obj_func(theta, batch_X, batch_y)-obj_func(theta_prev, batch_X, batch_y)) > threshold:
		t_start += batch_size
		t_stop += batch_size
		if t_start >= len(X):
			t_start %= len(X)
			t_stop %= len(X)
		batch_X = np.take(X, range(t_start, t_stop), mode = wrap)
		batch_y = np.take(y, range(t_start, t_stop), mode = wrap)
		theta_prev = theta
		theta = theta_prev - step_size * grad_func(theta_prev, batch_X, batch_y)
	return theta, obj_func(theta, batch_X, batch_y)

'''
Performs stochastic gradient descent procedure to find minimum value min of obj_func
and parameters theta_min such that obj_fun(theta_min)=min
data - n X d matrix containing n data vectors
init_theta - d X 1 column vector with initial guess for theta_min
obj_func - scalar function of data and theta
grad_func - vector function of data and theta, d X 1 column vector
step_size - scalar that multiplies gradient when updating theta
threshold - if successive values of the objective function differ by less than threshold,
	stop the gradient descent procedure

Returns - theta_min, obj_func(theta_min)
'''
def stochastic_gradient_descent(X, y, obj_func, grad_func, init_theta, learning_rate, threshold):
	t = 0
	theta_prev = init_theta
	theta = theta_prev - learning_rate(t) * grad_func(theta_prev, np.array([X[t]]), np.array([y[t]]))
	while abs(obj_func(theta, X, y)-obj_func(theta_prev, X, y)) > threshold:
		t += 1
		t %= len(X)
		theta_prev = theta
		theta = theta_prev - learning_rate(t) * grad_func(theta_prev, np.array([X[t]]), np.array([y[t]]))
	return theta, obj_func(theta, X, y)

def init_globals_gaussian(mean, cov):
	global neg_gaussian_mean, neg_gaussian_cov, neg_gaussian_coef, neg_gaussian_inv
	n = len(cov)
	neg_gaussian_mean = np.reshape(np.array(mean),(n,1))
	neg_gaussian_cov = np.array(cov)
	neg_gaussian_coef = -1.0/math.sqrt((2*math.pi)**n*np.linalg.det(cov))
	neg_gaussian_inv = np.linalg.inv(cov)

def init_globals_quad(A,b):
	global quad_A, quad_b
	n = len(b)
	quad_A = np.array(A)
	quad_b = np.reshape(np.array(b),(n,1))

def neg_gaussian(x):
	global neg_gaussian_coef, neg_gaussian_inv, neg_gaussian_mean, neg_gaussian_cov
	return neg_gaussian_coef*math.e**(-0.5*np.dot(np.dot(np.transpose(x-neg_gaussian_mean),neg_gaussian_inv),x-neg_gaussian_mean))

def neg_gaussian_deriv(x):
	global neg_gaussian_coef, neg_gaussian_inv, neg_gaussian_mean, neg_gaussian_cov
	return -1*neg_gaussian(x)*np.dot(neg_gaussian_inv, x-neg_gaussian_mean)

def quad_bowl(x):
	global quad_A, quad_b
	return 0.5*np.dot(np.dot(np.transpose(x),quad_A),x)-np.dot(np.transpose(x),quad_b)

def quad_bowl_deriv(x):
	global quad_A, quad_b
	return np.dot(quad_A,x)-quad_b

def least_square(theta, X, y):
	n = len(y)
	return sum(np.array([(np.dot(np.transpose(X[i]),theta)-y[i])**2 for i in range(n)]))

def least_square_deriv(theta, X, y):
	n = len(y)
	d = len(theta)
	gradient = np.array([[0.0] for i in range(d)])
	for i in range(n):
		gradient += 2*(int(np.dot(np.transpose(X[i]),theta))-y[i])*X[i]
	return gradient

def learning_rate(t):
	return (10**4+t)**(-1)

gaussMean,gaussCov,quadBowlA,quadBowlb = parameters.getData()

neg_gaussian_mean = 0
neg_gaussian_cov = 0
neg_gaussian_coef = 0
neg_gaussian_inv = 0
init_globals_gaussian(gaussMean, gaussCov)
print gradient_descent(neg_gaussian, neg_gaussian_deriv, [[0],[0]], 10**7, 10**(-10))

quad_A = 0
quad_b = 0
init_globals_quad(quadBowlA, quadBowlb)
print gradient_descent(quad_bowl, quad_bowl_deriv, [[-10],[-10]], 0.01, 10**(1))

print gradient_approx(quad_bowl, [0,0], 0.05)

X,y = fitting.getData()
X = np.array(X)
n, d = X.shape
X = np.array([np.reshape(X[i],(d,1)) for i in range(n)])
print batch_gradient_descent(X, y, least_square, least_square_deriv, [[0] for i in range(10)], 10**(-5), 10**(-1))

print stochastic_gradient_descent(X, y, least_square, least_square_deriv, [[0] for i in range(10)], learning_rate, 10**(-1))

