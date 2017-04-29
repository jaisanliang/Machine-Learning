import pdb
import sys

import math
import random
import numpy as np
import matplotlib.pyplot as plt

class ServerGenerator():
	def __init__(self):
		self.p = 0.06
		self.h = 0.5
		self.d = [0.5,0.16,0.17,0.17]
		self.free_servers = 10
		self.state = (0,1)
	def step(self,policy,current_state):
		action = policy[current_state].keys()[np.random.choice(len(policy[current_state].keys()),p=policy[current_state].values())]
		if action == -1 and self.free_servers > 0:
			reward = current_state[1]
		else:
			reward = 0
		self.free_servers = max(0,self.free_servers+action)
		for i in range(10-self.free_servers):
			if random.random() < self.p:
				self.free_servers += 1
		next_priority = np.random.choice([1,2,4,8],p=self.d)
		next_state = (self.free_servers,next_priority)
		return (action,next_state,reward)

def r_learning(states,actions,generator_class,generator_args,alpha=0.01,beta=0.01):
	epsilon = 0.1
	r = 0
	policy = {state: {action: 1.0/len(actions) for action in actions} for state in states}
	action_values = {(state,action): 0 for state in states for action in actions}
	generator = generator_class(*generator_args)
	current_state = generator.state
	for i in range(2000000):
		action, next_state, reward = generator.step(policy, current_state)
		next_action = action
		next_action_value = action_values[(next_state,action)]
		for a in actions:
			if action_values[(next_state,a)] > next_action_value:
				next_action_value = action_values[(next_state,a)]
				next_action = a
		delta = reward - r + action_values[(next_state,next_action)] - action_values[(current_state,action)]
		action_values[(current_state,action)] += alpha*delta
		best_action_value = action_values[(current_state,action)]
		for a in actions:
			if action_values[(current_state,a)] > best_action_value:
				best_action_value = action_values[(current_state,a)]
		if action_values[(current_state,action)] == best_action_value:
			r += beta*delta
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
	print r
	return policy, action_values

# replicate Figure 11.3
def server():
	states = [(i,j) for i in range(11) for j in [1,2,4,8]]
	actions = [0,-1]
	policy, action_values = r_learning(states,actions,ServerGenerator,[])
	for i in [1,2,4,8]:
		for j in range(11):
			sys.stdout.write(str(int(policy[(j,i)][0]<0.5))+' ')
		sys.stdout.write('\n')
	print [max(action_values[((i,1),0)],action_values[((i,1),-1)]) for i in range(11)]
	print [max(action_values[((i,2),0)],action_values[((i,2),-1)]) for i in range(11)]
	print [max(action_values[((i,4),0)],action_values[((i,4),-1)]) for i in range(11)]
	print [max(action_values[((i,8),0)],action_values[((i,8),-1)]) for i in range(11)]
	pdb.set_trace()

server()
