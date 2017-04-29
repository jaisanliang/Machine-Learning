import pdb
import sys

import math
import numpy as np
import matplotlib.pyplot as plt

def gradient_td_lambda_online(states, actions, generator_class, generator_args, l, alpha, v, grad, d):
	gamma = 1
	rms_error = np.zeros(100)
	for i in range(100):
		w = np.zeros(d)
		policies = {state: {action: 1.0/len(actions) for action in actions} for state in states}
		errors = []
		for j in range(20):
			z = np.zeros(d)
			generator = generator_class(*generator_args)
			current_state = generator.state
			while True:
				action, next_state, reward = generator.step(policies, current_state)
				if next_state == None:
					delta = reward-v(current_state,w)
				else:
					delta = reward+gamma*v(next_state,w)-v(current_state,w)
				z = gamma*l*z + grad(state,w)
				w += alpha*delta*z
				if next_state == None:
					break
				current_state = next_state
			errors.append(np.average([(values[state]-(state+1)/10.0+1)**2 for state in states])**0.5)
		rms_error[i] = np.average(errors)
	return np.average(rms_error)
