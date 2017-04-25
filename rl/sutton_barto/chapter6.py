import pdb
import sys

import math
import numpy as np
import matplotlib.pyplot as plt
import chapter5

class RandomWalkGenerator():
	def __init__(self):
		self.n = 5
		self.state = self.n/2
	def step(self, policy, current_state):
		action = np.random.choice(policy[current_state].keys(), p=policy[current_state].values())
		if current_state + action < 0:
			return (action,None,0)
		elif current_state + action >= self.n:
			return (action,None,1)
		else:
			return (action,current_state+action,0)

# constant-alpha MC
def constant_alpha_mc(states, actions, generator_class, alpha):
	gamma = 1
	rms_error = np.zeros(100)
	for i in range(100):
		values = {state: 0.5 for state in states}
		policies = {state: {action: 1.0/len(actions) for action in actions} for state in states}
		errors = []
		for j in range(100):
			episode_states = []
			generator = generator_class()
			current_state = generator.state
			while True:
				episode_states.append(current_state)
				action, next_state, reward = generator.step(policies, current_state)
				if next_state == None:
					break
				current_state = next_state
			for state in episode_states:
				values[state] += alpha*(reward-values[state])
			errors.append(np.average([(values[state]-(state+1)/6.0)**2 for state in states])**0.5)
		rms_error = rms_error+1.0/(i+1)*(np.array(errors)-rms_error)
	return rms_error

# tabular TD(0)
def tabular_td(states, actions, generator_class, alpha):
	gamma = 1
	rms_error = np.zeros(100)
	for i in range(100):
		values = {state: 0.5 for state in states}
		policies = {state: {action: 1.0/len(actions) for action in actions} for state in states}
		errors = []
		for j in range(100):
			generator = generator_class()
			current_state = generator.state
			while True:
				action, next_state, reward = generator.step(policies, current_state)
				if next_state == None:
					values[current_state] += alpha*(reward-values[current_state])
					break
				values[current_state] += alpha*(reward+gamma*values[next_state]-values[current_state])
				current_state = next_state
			errors.append(np.average([(values[state]-(state+1)/6.0)**2 for state in states])**0.5)
		rms_error = rms_error+1.0/(i+1)*(np.array(errors)-rms_error)
	return rms_error

# replicate Figure 6.7
def random_walk(n):
	error15 = tabular_td(range(n),[-1,1],RandomWalkGenerator,0.15)
	error10 = tabular_td(range(n),[-1,1],RandomWalkGenerator,0.1)
	error05 = tabular_td(range(n),[-1,1],RandomWalkGenerator,0.05)
	error03 = constant_alpha_mc(range(n),[-1,1],RandomWalkGenerator,0.03)
	plt.plot(range(100),error15,'r',range(100),error10,'b',range(100),error05,'g',range(100),error03,'k')
	plt.show()

class WindyGridwalkGenerator():
	def __init__(self, isStochastic = False):
		self.state = (0,3)
		self.isStochastic = isStochastic
	def step(self, policy, current_state):
		action = policy[current_state].keys()[np.random.choice(len(policy[current_state].keys()), p=policy[current_state].values())]
		# apply wind
		stochasticOffset = self.isStochastic*np.random.choice([-1,0,1])
		if current_state[0] in [3,4,5,8]:
			wind_action = (action[0],action[1]+1)
		elif current_state[0] in [6,7]:
			wind_action = (action[0],action[1]+2)
		else:
			wind_action = action
		wind_action = (wind_action[0],wind_action[1]+stochasticOffset)
		current_state = (min(max(0,current_state[0]+wind_action[0]),9), min(max(0,current_state[1]+wind_action[1]),6))
		if current_state == (7,3):
			return (action,None,-1)
		else:
			return (action,current_state,-1)

# SARSA
def sarsa(states, actions, generator_class, generator_args, alpha):
	gamma = 1
	epsilon = 0.1
	action_values = {(state,action): 0 for state in states for action in actions}
	policies = {state: {action: 1.0/len(actions) for action in actions} for state in states}
	steps = 0
	for j in range(170):
		generator = generator_class(*generator_args)
		current_state = generator.state
		num_steps = 0
		while True:
			num_steps += 1
			action, next_state, reward = generator.step(policies, current_state)
			if next_state == None:
				action_values[(current_state,action)] += alpha*(reward-action_values[(current_state,action)])
				break
			next_action = policies[next_state].keys()[np.random.choice(len(policies[next_state].keys()), p=policies[next_state].values())]
			action_values[(current_state,action)] += alpha*(reward+gamma*action_values[(next_state,next_action)]-action_values[(current_state,action)])
			current_state = next_state
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
		print num_steps
		steps += num_steps
	print steps
	return action_values, policies

# replicate Figure 6.11
def windy_gridworld():
	states = [(i,j) for i in range(10) for j in range(7)]
	actions = [(-1,0),(1,0),(0,-1),(0,1)]
	action_values, policies = sarsa(states,actions,WindyGridwalkGenerator,[False],0.5)

# Problem 6.6
def windy_gridworld_king():
	states = [(i,j) for i in range(10) for j in range(7)]
	# actions = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
	actions = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(0,0)]
	action_values, policies = sarsa(states,actions,WindyGridwalkGenerator,[False],0.5)

# Problem 6.7
def windy_gridworld_king_stochastic():
	states = [(i,j) for i in range(10) for j in range(7)]
	actions = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
	action_values, policies = sarsa(states,actions,WindyGridwalkGenerator,[True],0.5)

class CliffGenerator():
	def __init__(self):
		self.state = (0,0)
	def step(self, policy, current_state):
		action = policy[current_state].keys()[np.random.choice(len(policy[current_state].keys()),p=policy[current_state].values())]
		next_state = (min(11,max(0,current_state[0]+action[0])),min(3,max(0,current_state[1]+action[1])))
		if next_state == (11,0):
			return (action,None,-1)
		elif next_state[1] == 0 and (next_state[0] > 0 and next_state[0] < 11):
			return (action,(0,0),-100)
		else:
			return (action,next_state,-1)

# q-learning
def q_learning(states, actions, generator_class, generator_args, alpha):
	gamma = 1
	epsilon = 0.1
	action_values = {(state,action): 0 for state in states for action in actions}
	policy = {state: {action: 1.0/len(actions) for action in actions} for state in states}
	steps = 0
	for i in range(170):
		#if i % 10 == 0:
		#	print i
		num_steps = 0
		generator = generator_class(*generator_args)
		current_state = generator.state
		while True:
			num_steps += 1
			action, next_state, reward = generator.step(policy,current_state)
			if next_state == None:
				action_values[(current_state,action)] += alpha*(reward-action_values[(current_state,action)])
				break
			next_action = max(policy[next_state],key=policy[next_state].get)
			action_values[(current_state,action)] += alpha*(reward+gamma*action_values[(next_state,next_action)]-action_values[(current_state,action)])
			current_state = next_state
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
		print num_steps
		steps += num_steps
	print steps
	return action_values, policy

# replicate Figure 6.13
def cliff():
	states = [(i,j) for i in range(12) for j in range(4)]
	actions = [(-1,0),(1,0),(0,-1),(0,1)]
	action_values, policy = sarsa(states, actions, CliffGenerator, [], 0.5)
