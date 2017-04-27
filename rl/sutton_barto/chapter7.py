import pdb
import sys

import math
import numpy as np
import matplotlib.pyplot as plt
import chapter6

class RandomWalkGenerator():
	def __init__(self,num_states):
		self.num_states = num_states
		self.state = self.num_states/2
	def step(self, policy, current_state):
		action = np.random.choice(policy[current_state].keys(), p=policy[current_state].values())
		if current_state + action < 0:
			return (action,None,-1)
		elif current_state + action >= self.num_states:
			return (action,None,1)
		else:
			return (action,current_state+action,0)

# tabular offline TD(n) evaluation
def tabular_td_n_offline(states, actions, generator_class, generator_args, n, alpha):
	gamma = 1
	rms_error = np.zeros(100)
	for i in range(100):
		values = {state: 0 for state in states}
		policies = {state: {action: 1.0/len(actions) for action in actions} for state in states}
		errors = []
		for j in range(10):
			episode_states = []
			rewards = []
			generator = generator_class(*generator_args)
			current_state = generator.state
			while True:
				action, next_state, reward = generator.step(policies, current_state)
				episode_states.append(current_state)
				rewards.append(reward)
				if next_state == None:
					break
				current_state = next_state
			# offline returns
			new_values = {state: values[state] for state in states}
			for t, state in enumerate(episode_states):
				returns = 0
				for t_s in range(n):
					if t+t_s < len(episode_states):
						returns += gamma**t_s*rewards[t+t_s]
				if t+n < len(episode_states):
					last_episode_value = values[episode_states[t+n]]
				else:
					last_episode_value = 0
				new_values[state] += alpha*(returns+last_episode_value-values[state])
			values = new_values
			errors.append(np.average([(values[state]-(state+1)/10.0+1)**2 for state in states])**0.5)
		rms_error[i] = np.average(errors)
	return np.average(rms_error)

# tabular online TD(n) evaluation
def tabular_td_n_online(states, actions, generator_class, generator_args, n, alpha):
	gamma = 1
	rms_error = np.zeros(100)
	for i in range(100):
		values = {state: 0 for state in states}
		policies = {state: {action: 1.0/len(actions) for action in actions} for state in states}
		errors = []
		for j in range(10):
			episode_states = []
			rewards = []
			generator = generator_class(*generator_args)
			current_state = generator.state
			while True:
				action, next_state, reward = generator.step(policies, current_state)
				episode_states.append(current_state)
				rewards.append(reward)
				if next_state == None:
					break
				current_state = next_state
			# online returns
			for t, state in enumerate(episode_states):
				returns = 0
				for t_s in range(n):
					if t+t_s < len(episode_states):
						returns += gamma**t_s*rewards[t+t_s]
				if t+n < len(episode_states):
					last_episode_value = values[episode_states[t+n]]
				else:
					last_episode_value = 0
				values[state] += alpha*(returns+last_episode_value-values[state])
			errors.append(np.average([(values[state]-(state+1)/10.0+1)**2 for state in states])**0.5)
		rms_error[i] = np.average(errors)
	return np.average(rms_error)

# tabular offline TD(lambda) evaluation
def tabular_td_lambda_offline(states, actions, generator_class, generator_args, l, alpha):
	gamma = 1
	rms_error = np.zeros(100)
	for i in range(100):
		values = {state: 0 for state in states}
		policies = {state: {action: 1.0/len(actions) for action in actions} for state in states}
		errors = []
		for j in range(10):
			episode_states = []
			rewards = []
			generator = generator_class(*generator_args)
			current_state = generator.state
			while True:
				action, next_state, reward = generator.step(policies, current_state)
				episode_states.append(current_state)
				rewards.append(reward)
				if next_state == None:
					break
				current_state = next_state
			# offline returns
			new_values = {state: values[state] for state in states}
			z = {state: 0 for state in states}
			for t, state in enumerate(episode_states):
				z[state] += 1
				if t < len(episode_states) - 1:
					delta = rewards[t]+gamma*values[episode_states[t+1]]-values[state]
				else:
					delta = rewards[t]-values[state]
				for state in states:
					new_values[state] += alpha*delta*z[state]
					z[state] *= (gamma*l)
			values = new_values
			errors.append(np.average([(values[state]-(state+1)/10.0+1)**2 for state in states])**0.5)
		rms_error[i] = np.average(errors)
	return np.average(rms_error)

# tabular online TD(lambda) evaluation
def tabular_td_lambda_online(states, actions, generator_class, generator_args, l, alpha):
	gamma = 1
	rms_error = np.zeros(100)
	for i in range(100):
		values = {state: 0 for state in states}
		policies = {state: {action: 1.0/len(actions) for action in actions} for state in states}
		errors = []
		for j in range(20):
			z = {state: 0 for state in states}
			generator = generator_class(*generator_args)
			current_state = generator.state
			while True:
				action, next_state, reward = generator.step(policies, current_state)
				z[current_state] += 1
				if next_state == None:
					delta = reward-values[current_state]
				else:
					delta = reward+gamma*values[next_state]-values[current_state]
				for state in states:
					values[state] += alpha*delta*z[state]
					z[state] *= (gamma*l)
				if next_state == None:
					break
				current_state = next_state
			errors.append(np.average([(values[state]-(state+1)/10.0+1)**2 for state in states])**0.5)
		rms_error[i] = np.average(errors)
	return np.average(rms_error)

def sarsa_lambda(states, actions, generator_class, generator_args, l, alpha, accumulating_trace = True):
	gamma = 1
	epsilon = 0.1
	action_values = {(state,action): 0 for state in states for action in actions}
	policies = {state: {action: 1.0/len(actions) for action in actions} for state in states}
	steps = 0
	for j in range(100):
		z = {(state,action): 0 for state in states for action in actions}
		generator = generator_class(*generator_args)
		current_state = generator.state
		num_steps = 0
		while True:
			num_steps += 1
			action, next_state, reward = generator.step(policies, current_state)
			if accumulating_trace:
				z[(current_state,action)] += 1
			else:
				z[(current_state,action)] = 1
			if next_state == None:
				delta = reward-action_values[(current_state,action)]
			else:
				next_action = policies[next_state].keys()[np.random.choice(len(policies[next_state].keys()), p=policies[next_state].values())]
				delta = reward+gamma*action_values[(next_state,next_action)]-action_values[(current_state,action)]
			for action_state in action_values:
				action_values[action_state] += alpha*delta*z[action_state]
				z[action_state] *= (gamma*l)
			# update policy
			for state in policies:
				best_action = None
				best_action_value = -float('inf')
				for action in actions:
					if action_values[(state,action)] > best_action_value:
						best_action = action
						best_action_value = action_values[(state,action)]
				policies[state] = {action: epsilon/len(actions) for action in actions}
				policies[state][best_action] += 1-epsilon
			if next_state == None:
				break
			current_state = next_state
		#print num_steps
		steps += num_steps
	print steps
	return action_values, policies

def q_lambda(states, actions, generator_class, generator_args, l, alpha):
	gamma = 1
	epsilon = 0.1
	action_values = {(state,action): 0 for state in states for action in actions}
	policy = {state: {action: 1.0/len(actions) for action in actions} for state in states}
	for j in range(1000):
		z = {(state,action): 0 for state in states for action in actions}
		generator = generator_class(*generator_args)
		current_state = generator.state
		while True:
			action, next_state, reward = generator.step(policy, current_state)
			z[(current_state,action)] += 1
			if next_state == None:
				delta = reward-action_values[(current_state,action)]
			else:
				next_action = policy[next_state].keys()[np.random.choice(len(policy[next_state].keys()), p=policy[next_state].values())]
				best_action = next_action
				for a in actions:
					if action_values[(next_state,a)] > action_values[(next_state,best_action)]:
						best_action = a
				delta = reward+gamma*action_values[(next_state,best_action)]-action_values[(current_state,action)]
			for action_state in action_values:
				action_values[action_state] += alpha*delta*z[action_state]
				if best_action == next_action:
					z[action_state] *= (gamma*l)
				else:
					z[action_state] = 0
			if next_state == None:
				break
			# update policy
			for state in policy:
				best_action = None
				best_action_value = -float('inf')
				for action in actions:
					if action_values[(state,action)] > best_action_value:
						best_action = action
						best_action_value = action_values[(state,action)]
				policy[state] = {action: epsilon/len(actions) for action in actions}
				policy[state][best_action] += 1-epsilon
			
			current_state = next_state
	return action_values, policy

# tabular online TD(lambda) evaluation with replacing traces
def tabular_td_lambda_online_replacing_traces(states, actions, generator_class, generator_args, l, alpha):
	gamma = 1
	rms_error = np.zeros(100)
	for i in range(100):
		values = {state: 0 for state in states}
		policies = {state: {action: 1.0/len(actions) for action in actions} for state in states}
		errors = []
		for j in range(20):
			z = {state: 0 for state in states}
			generator = generator_class(*generator_args)
			current_state = generator.state
			while True:
				action, next_state, reward = generator.step(policies, current_state)
				z[current_state] = 1
				if next_state == None:
					delta = reward-values[current_state]
				else:
					delta = reward+gamma*values[next_state]-values[current_state]
				for state in states:
					values[state] += alpha*delta*z[state]
					z[state] *= (gamma*l)
				if next_state == None:
					break
				current_state = next_state
			errors.append(np.average([(values[state]-(state+1)/10.0+1)**2 for state in states])**0.5)
		rms_error[i] = np.average(errors)
	return np.average(rms_error)

def random_walk():
	#print tabular_td_n(range(19),[-1,1],RandomWalkGenerator,[19],8,0.1)
	#return
	'''
	# replicate Figure 7.2
	errors = {8:[]}
	alphas = [0.0,0.05,0.1,0.15,0.2,0.25,0.3]
	for n in errors.keys():
		for a in alphas:
			errors[n].append(tabular_td_n_offline(range(19),[-1,1],RandomWalkGenerator,[19],n,a))
	plt.plot(alphas,errors[8])
	plt.show()
	
	# replicate Figure 7.2
	errors = {1:[]}
	alphas = [0.0,0.2,0.4,0.6,0.8]
	for n in errors.keys():
		for a in alphas:
			errors[n].append(tabular_td_n_online(range(19),[-1,1],RandomWalkGenerator,[19],n,a))
	plt.plot(alphas,errors[1])
	plt.show()
	'''
	'''
	# replicate Figure 7.6
	errors = {0.4:[], 0.8:[]}
	alphas = [0.0,0.05,0.1,0.15,0.2]
	for l in errors.keys():
		for a in alphas:
			errors[l].append(tabular_td_lambda_offline(range(19),[-1,1],RandomWalkGenerator,[19],l,a))
	plt.plot(alphas,errors[0.4],alphas,errors[0.8])
	plt.show()
	'''
	'''
	# replicate Figure 7.9
	errors = {0.4:[], 0.6:[]}
	alphas = [0.0,0.2,0.4,0.6,0.8,1]
	for l in errors.keys():
		for a in alphas:
			errors[l].append(tabular_td_lambda_online(range(19),[-1,1],RandomWalkGenerator,[19],l,a))
	plt.plot(alphas,errors[0.4],alphas,errors[0.6])
	plt.show()
	'''

	# replicate Figure 7.17
	errors = {0.9:[]}
	alphas = [0.0,0.1,0.2,0.3,0.4,0.5]
	for l in errors.keys():
		for a in alphas:
			# errors[l].append(tabular_td_lambda_online(range(19),[-1,1],RandomWalkGenerator,[19],l,a))
			errors[l].append(tabular_td_lambda_online_replacing_traces(range(19),[-1,1],RandomWalkGenerator,[19],l,a))
	plt.plot(alphas,errors[0.9])
	plt.show()

class GraphGenerator():
	def __init__(self):
		self.state = 0
	def step(self,policy,current_state):
		action = np.random.choice(policy[current_state].keys(), p=policy[current_state].values())
		if current_state + action > 4:
			return (action,None,1)
		else:
			return (action,current_state+action,0)

def cliff():
	states = [(i,j) for i in range(12) for j in range(4)]
	actions = [(-1,0),(1,0),(0,-1),(0,1)]
	action_values, policy = q_lambda(states, actions, chapter6.CliffGenerator, [], 0.4, 0.5)

# Problem 7.6
def graph():
	states = range(5)
	actions = [0,1]
	action_values, policy = sarsa_lambda(states, actions, GraphGenerator, [], 0.9, 0.3)
	action_values, policy = sarsa_lambda(states, actions, GraphGenerator, [], 0.9, 0.3, accumulating_trace=False)

graph()
