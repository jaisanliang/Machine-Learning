import math
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
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
	print "Starting gradient descent. Initial theta: " + str(np.transpose(init_theta)) + ". Step size " + str(step_size) + " Threshold: " + str(threshold)
	theta_prev = init_theta
	theta = theta_prev - step_size * grad_func(theta_prev)
	count = 0
	gradnorm = []
	while abs(obj_func(theta)-obj_func(theta_prev)) > threshold and count < 1000:
		print "Theta: " + str(np.transpose(theta)) + ". Obj Func: " + str(obj_func(theta))
		theta_prev = theta
		theta = theta_prev - step_size * grad_func(theta_prev)
		gradnorm.append(np.linalg.norm(grad_func(theta_prev)))
		count += 1
	return theta, obj_func(theta), count, gradnorm

'''
Central difference approximation for the gradient of a function
obj_func - scalar function of theta
theta - d X 1 column vector
delta - scalar controlling the difference step size

Returns - d X 1 gradient vector
'''
def gradient_approx(obj_func, theta, delta):
#	return [obj_func(theta-np.array([[delta/2*(i==j)] for j in range(len(theta))])) for i in range(len(theta))]	
	return np.array([obj_func(theta+np.array([[delta/2*(i==j)] for j in range(len(theta))]))-obj_func(theta-np.array([[delta/2*(i==j)] for j in range(len(theta))])) for i in range(len(theta))])/delta

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
	hist = []
	while abs(obj_func(theta, X, y)-obj_func(theta_prev, X, y)) > threshold:
		hist.append(obj_func(theta, X, y))
		theta_prev = theta
		theta = theta_prev - step_size * grad_func(theta_prev, X, y)
	return theta, obj_func(theta, X, y), hist

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
	hist = []
	while abs(obj_func(theta, X, y)-obj_func(theta_prev, X, y)) > threshold:
		hist.append(obj_func(theta, X, y))
		t += 1
		t %= len(X)
		theta_prev = theta
		theta = theta_prev - learning_rate(t) * grad_func(theta_prev, np.array([X[t]]), np.array([y[t]]))
	return theta, obj_func(theta, X, y), hist

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


def plot_iter(func, func_deriv, init_theta, step_size, threshold):
	x_n=[10**i for i in range (-5,5)]
	l_n=[i for i in range(-5,5)]
	y_n=[np.linalg.norm(gradient_descent(func,func_deriv,init_theta,step_size,x)[0]-[[26.67],[26.67]]) for x  in x_n]
	plt.plot(l_n,y_n)
	plt.title('Quadratic Bowl')
	plt.xlabel('log(threshold)')
	plt.ylabel('norm(diff)')
	plt.show()
	y_n=[gradient_descent(func,func_deriv,init_theta,step_size,x)[2] for x  in x_n]
	plt.plot(l_n,y_n)
	plt.title('Quadratic Bowl')
	plt.xlabel('log(threshold)')
	plt.ylabel('iterations')
	plt.show()

gaussMean,gaussCov,quadBowlA,quadBowlb = parameters.getData()

neg_gaussian_mean = 0
neg_gaussian_cov = 0
neg_gaussian_coef = 0
neg_gaussian_inv = 0
init_globals_gaussian(gaussMean, gaussCov)
#print "Running gradient descent on negative gaussian:"
#plot_iter(neg_gaussian, neg_gaussian_deriv, [[0],[0]],10**7,10**(-10))
#print gradient_descent(neg_gaussian, neg_gaussian_deriv, [[-1],[-1]],10**7, 10**(-10))[2]

#print "Running gradient approx on negative gaussian:"
quad_A = 0
quad_b = 0
init_globals_quad(quadBowlA, quadBowlb)
'''
print "-10,10:"
print quad_bowl_deriv([[-10],[10]])
print gradient_approx(quad_bowl,[[-10],[10]],.1) 
print gradient_approx(quad_bowl,[[-10],[10]],.5) 
print gradient_approx(quad_bowl,[[-10],[10]],5)

print "100,100: "
print quad_bowl_deriv([[100],[100]])
print gradient_approx(quad_bowl,[[100],[100]],.1) 
print gradient_approx(quad_bowl,[[100],[100]],.5) 
print gradient_approx(quad_bowl,[[100],[100]],5)

print "32, 34:"
print quad_bowl_deriv([[32],[-34]])
print gradient_approx(quad_bowl,[[32],[-34]],.1) 
print gradient_approx(quad_bowl,[[32],[-34]],.5) 
print gradient_approx(quad_bowl,[[32],[-34]],5)
 

#print "Running gradient descent on quadratic bowl:"
#plot_iter(quad_bowl,quad_bowl_deriv,[[-13],[-12]],10**(-3),10**(1))
#y = gradient_descent(quad_bowl, quad_bowl_deriv, [[-10],[-10]], 10**(-1), 10**(-15))[3]
#print y
#plt.plot(y)
#plt.show()
'''
print "Running gradient approx on quadratic bowl:"
print gradient_approx(quad_bowl, [0,0], 0.05)

X,y = fitting.getData()
X = np.array(X)
n, d = X.shape
X = np.array([np.reshape(X[i],(d,1)) for i in range(n)])
print "Running batch gradient descent:"
batch = batch_gradient_descent(X, y, least_square, least_square_deriv, [[0] for i in range(10)], 10**(-5), 10**(-1))
xb = [i for i in range(len(batch[2]))]
print batch
print len(xb)
print "Running stochastic gradient descent:"
stoch = stochastic_gradient_descent(X, y, least_square, least_square_deriv, [[0] for i in range(10)], learning_rate, 10**(-1)) 
xs = [i for i in range(len(stoch[2]))]
print len(xs)
print stoch
plt.plot(xs,stoch[2],xb,batch[2])
plt.axis([1,80,8000,1000000])
plt.show()
