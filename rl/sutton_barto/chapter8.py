import pdb
import sys

import math
import numpy as np
import matplotlib.pyplot as plt

class BlockingGenerator():
	def __init__(self):
		self.state = (3,0)
	def step(self, policy, current_state):
		action = policy[current_state].keys()[np.random.choice(len(policy[current_state].keys()),p=policy[current_state].values())]
		proposed_state = (min(8,max(0,current_state[0]+action[0])),min(5,max(0,current_state[1]+action[1])))
		# agent can't run into walls
		if proposed_state[1] == 2 and proposed_state[0] < 8:
			return (action,current_state,0)
		elif proposed_state[1] == 5 and proposed_state[0] == 8:
			return (action,(3,0),1)
		else:
			return (action,proposed_state,0)

def dyna_q(states, actions, generator_class, generator_args, n):
	alpha = 0.1
	gamma = 0.95
	epsilon = 0.1
	action_values = {(state,action): 1 for state in states for action in actions}
	model = {}
	policy = {state: {action: 1.0/len(actions) for action in actions} for state in states}
	num_steps = 0
	cum_reward = 0
	while num_steps < 1000:
		generator = generator_class(*generator_args)
		current_state = generator.state
		while num_steps < 1000:
			num_steps += 1
			action, next_state, reward = generator.step(policy, current_state)
			cum_reward += reward
			if next_state == None:
				action_values[(current_state,action)] += alpha*(reward-action_values[(current_state,action)])
			else:
				next_action = max(policy[next_state],key=policy[next_state].get)
				action_values[(current_state,action)] += alpha*(reward+gamma*action_values[(next_state,next_action)]-action_values[(current_state,action)])
			model[(current_state,action)] = (reward,next_state)
			current_state = next_state
			for i in range(n):
				action_state = model.keys()[np.random.randint(len(model.keys()))]
				reward, next_state = model[action_state]
				next_action = max(policy[next_state],key=policy[next_state].get)
				action_values[action_state] += alpha*(reward+gamma*action_values[(next_state,next_action)]-action_values[action_state])
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
	print cum_reward
	return action_values, policy

def dyna_q_plus(states, actions, generator_class, generator_args, n):
	alpha = 0.1
	gamma = 0.95
	epsilon = 0.1
	kappa = 0.01
	action_values = {(state,action): 1 for state in states for action in actions}
	tau = {(state,action): 0 for state in states for action in actions}
	model = {}
	policy = {state: {action: 1.0/len(actions) for action in actions} for state in states}
	num_steps = 0
	cum_reward = 0
	while num_steps < 1000:
		generator = generator_class(*generator_args)
		current_state = generator.state
		while num_steps < 1000:
			num_steps += 1
			action, next_state, reward = generator.step(policy, current_state)
			cum_reward += reward
			if next_state == None:
				action_values[(current_state,action)] += alpha*(reward-action_values[(current_state,action)])
			else:
				next_action = max(policy[next_state],key=policy[next_state].get)
				action_values[(current_state,action)] += alpha*(reward+gamma*action_values[(next_state,next_action)]-action_values[(current_state,action)])
			model[(current_state,action)] = (reward,next_state)
			tau = {(state,action): tau[(state,action)]+1 for state in states for action in actions}
			tau[(current_state,action)] = 0
			current_state = next_state
			for i in range(n):
				action_state = model.keys()[np.random.randint(len(model.keys()))]
				reward, next_state = model[action_state]
				next_action = max(policy[next_state],key=policy[next_state].get)
				action_values[action_state] += alpha*(reward+gamma*action_values[(next_state,next_action)]-action_values[action_state])
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
	print cum_reward
	return action_values, policy

def blocking():
	states = [(i,j) for i in range(9) for j in range(6)]
	actions = [(-1,0),(1,0),(0,-1),(0,1)]
	action_values, policy = dyna_q(states, actions, BlockingGenerator, [], 50)
	pdb.set_trace()

blocking()
