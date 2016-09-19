import numpy as np
import loadFittingDataP1 as fitting
import loadParametersP1 as parameters

def gradient_descent(obj_func, grad_func, init_theta, step_size, threshold):
	theta_prev = init_guess
	theta = theta_prev - step_size * grad_func(theta_prev)
	while abs(obj_func(theta)-obj_func(theta_prev)) > threshold:
		theta_prev = theta
		theta = theta_prev - step_size * grad_func(theta_prev)
	return theta, obj_func(theta)

def gradient_approx(obj_func, theta, delta):
	return np.array([obj_func(theta+np.array([delta/2*(i==j) for j in range(len(theta))]))-obj_func(theta-np.array([delta/2*(i==j) for j in range(len(theta))])) for i in range(len(theta))])/delta

def batch_gradient_descent(data, obj_func, grad_func, init_theta, step_size, threshold):
	theta_prev = init_theta
	theta = theta_prev - step_size * grad_func(data, theta_prev)
	while abs(obj_func(data, theta)-obj_func(data, theta_prev)) > threshold:
		theta_prev = theta
		theta = theta_prev - step_size * grad_func(data, theta_prev)
	return theta, obj_func(data, theta)

def mini_batch_gradient_descent(data, batch_size, obj_func, grad_func, init_theta, step_size, threshold):
	t_start = 0
	t_stop = batch_size
	theta_prev = init_theta
	batch = np.take(data, range(t_start, t_stop), mode = wrap)
	theta = theta_prev - step_size * grad_func(batch, theta_prev)
	while abs(obj_func(data, theta)-obj_func(data, theta_prev)) > threshold:
		t_start += batch_size
		t_stop += batch_size
		if t_start >= len(data):
			t_start %= len(data)
			t_stop %= len(data)
		batch = np.take(data, range(t_start, t_stop), mode = wrap)
		theta_prev = theta
		theta = theta_prev - step_size * grad_func(batch, theta_prev)
	return theta, obj_func(data, theta)

def stochastic_gradient_descent(data, obj_func, grad_func, init_theta, learning_rate, threshold):
	t = 0
	theta_prev = init_theta
	theta = theta_prev - step_size * grad_func(data[t], theta_prev)
	while abs(obj_func(data, theta)-obj_func(data, theta_prev)) > threshold:
		t += 1
		t %= len(data)
		theta_prev = theta
		theta = theta_prev - step_size * grad_func(data[t], theta_prev)
	return theta, obj_func(data, theta)

gaussMean,gaussCov,quadBowlA,quadBowlb = parameters.getData()

def neg_gaussian(x, mean, cov):
	n = len(mean)
	return -1.0/math.sqrt((2*math.pi)**n*np.linalg.det(cov))*math.e**(-1.0*np.dot(np.dot(np.transpose(x-mean),np.linalg.inverse(cov)),x-mean))
