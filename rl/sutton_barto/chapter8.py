import pdb
import sys

import math
import numpy as np
import matplotlib.pyplot as plt

class BlockingGenerator():
	def __init__(self):
		self.steps = 0
		self.state = (3,0)
	def step(self, policy, current_state):
		self.steps += 1
		action = policy[current_state].keys()[np.random.choice(len(policy[current_state].keys()),p=policy[current_state].values())]
		proposed_state = (min(8,max(0,current_state[0]+action[0])),min(5,max(0,current_state[1]+action[1])))
		# agent can't run into walls
		if proposed_state[1] == 2 and proposed_state[0] < 8 and self.steps < 1000:
			return (action,current_state,0)
		elif proposed_state[1] == 2 and proposed_state[0] > 0 and self.steps >= 1000:
			return (action,current_state,0)
		elif proposed_state[1] == 5 and proposed_state[0] == 8:
			return (action,(3,0),1)
		else:
			return (action,proposed_state,0)

class MazeGenerator():
	def __init__(self):
		self.state = (0,3)
	def step(self, policy, current_state):
		action = policy[current_state].keys()[np.random.choice(len(policy[current_state].keys()),p=policy[current_state].values())]
		proposed_state = (min(8,max(0,current_state[0]+action[0])),min(5,max(0,current_state[1]+action[1])))
		# agent can't run into walls
		if proposed_state in [(2,2),(2,3),(2,4),(5,1),(7,3),(7,4),(7,5)]:
			return (action,current_state,0)
		elif proposed_state == (8,5):
			return (action,None,1)
		else:
			return (action,proposed_state,0)

def dyna_q(states, actions, generator_class, generator_args, n):
	alpha = 0.1
	gamma = 0.95
	epsilon = 0.1
	episode_steps = np.zeros(5)
	episode_rewards = np.zeros(2000)
	for trial in range(10):
		action_values = {(state,action): 0 for state in states for action in actions}
		model = {}
		policy = {state: {action: 1.0/len(actions) for action in actions} for state in states}
		steps = []
		rewards = []
		cum_reward = 0
		for i in range(1):
			num_steps = 0
			generator = generator_class(*generator_args)
			current_state = generator.state
			#while True:
			for k in range(2000):
				num_steps += 1
				action, next_state, reward = generator.step(policy, current_state)
				cum_reward += reward
				rewards.append(cum_reward)
				if next_state == None:
					action_values[(current_state,action)] += alpha*(reward-action_values[(current_state,action)])
				else:
					max_value = max(action_values[(next_state,a)] for a in actions)
					possible_actions = [a for a in actions if action_values[(next_state,a)] > (max_value-0.01)]
					next_action = possible_actions[np.random.randint(len(possible_actions))]
					action_values[(current_state,action)] += alpha*(reward+gamma*action_values[(next_state,next_action)]-action_values[(current_state,action)])
				model[(current_state,action)] = (reward,next_state)
				current_state = next_state
				if next_state == None:
					break
				for j in range(n):
					action_state = model.keys()[np.random.randint(len(model.keys()))]
					reward, next_state = model[action_state]
					if next_state != None:
						next_action = max(policy[next_state],key=policy[next_state].get)
						action_values[action_state] += alpha*(reward+gamma*action_values[(next_state,next_action)]-action_values[action_state])
				# update policy
				for state in policy:
					best_action = None
					best_action_value = -float('inf')
					for action in actions:
						if action_values[(state,action)] > best_action_value:
							best_action = action
							best_action_value = action_values[(state,action)]
					if best_action_value > 0:
						policy[state] = {action: epsilon/len(actions) for action in actions}
						policy[state][best_action] += 1-epsilon
					else:
						policy[state] = {action: 1.0/len(actions) for action in actions}
			steps.append(num_steps)
		# episode_steps = 1.0*(episode_steps*trial+np.array(steps))/(trial+1)
		episode_rewards = 1.0*(episode_rewards*trial+np.array(rewards))/(trial+1)
	
	#return action_values, policy, episode_steps
	return action_values, policy, episode_rewards

def dyna_q_plus(states, actions, generator_class, generator_args, n):
	alpha = 0.1
	gamma = 0.95
	epsilon = 0.1
	kappa = 0.0001
	episode_steps = np.zeros(5)
	episode_rewards = np.zeros(2000)
	for trial in range(10):
		cum_reward = 0
		rewards = []
		action_values = {(state,action): 0 for state in states for action in actions}
		last_tried = {(state,action): 0 for state in states for action in actions}
		model = {}
		policy = {state: {action: 1.0/len(actions) for action in actions} for state in states}
		steps = []
		for i in range(1):
			num_steps = 0
			generator = generator_class(*generator_args)
			current_state = generator.state
			#while True:
			for k in range(2000):
				num_steps += 1
				action, next_state, reward = generator.step(policy, current_state)
				last_tried = {action_state: last_tried[action_state]+1 for action_state in last_tried}
				last_tried[(current_state,action)] = 0
				cum_reward += reward
				rewards.append(cum_reward)
				if next_state == None:
					action_values[(current_state,action)] += alpha*(reward-action_values[(current_state,action)])
				else:
					max_value = max(action_values[(next_state,a)] for a in actions)
					possible_actions = [a for a in actions if action_values[(next_state,a)] > (max_value-0.01)]
					next_action = possible_actions[np.random.randint(len(possible_actions))]
					action_values[(current_state,action)] += alpha*(reward+gamma*action_values[(next_state,next_action)]-action_values[(current_state,action)])
				model[(current_state,action)] = (reward,next_state)
				current_state = next_state
				if next_state == None:
					break
				for j in range(n):
					action_state = model.keys()[np.random.randint(len(model.keys()))]
					reward, next_state = model[action_state]
					if next_state != None:
						#next_action = max(policy[next_state],key=policy[next_state].get)
						#action_values[action_state] += alpha*(reward+gamma*action_values[(next_state,next_action)]-action_values[action_state])
						max_value = max(action_values[(next_state,a)]+kappa*last_tried[(next_state,a)]**0.5 for a in actions)
						possible_actions = [a for a in actions if action_values[(next_state,a)]+kappa*last_tried[(next_state,a)]**0.5 > (max_value-0.01)]
						next_action = possible_actions[np.random.randint(len(possible_actions))]
						action_values[action_state] += alpha*(reward+gamma*action_values[(next_state,next_action)]+gamma*kappa*last_tried[(next_state,next_action)]**0.5-action_values[action_state])
				# update policy
				for state in policy:
					best_action = None
					best_action_value = -float('inf')
					for action in actions:
						if action_values[(state,action)] > best_action_value:
							best_action = action
							best_action_value = action_values[(state,action)]
					if best_action_value > 0:
						policy[state] = {action: epsilon/len(actions) for action in actions}
						policy[state][best_action] += 1-epsilon
					else:
						policy[state] = {action: 1.0/len(actions) for action in actions}
			steps.append(num_steps)
		#episode_steps = 1.0*(episode_steps*trial+np.array(steps))/(trial+1)
		episode_rewards = 1.0*(episode_rewards*trial+np.array(rewards))/(trial+1)
	
	#return action_values, policy, episode_steps
	return action_values, policy, episode_rewards

from Queue import PriorityQueue

def dyna_q_proiritized_sweeping(states, actions, generator_class, generator_args, n):
	alpha = 0.1
	gamma = 0.95
	epsilon = 0.1
	theta = 0.1
	episode_steps = np.zeros(5)
	episode_rewards = np.zeros(2000)
	for trial in range(10):
		action_values = {(state,action): 0 for state in states for action in actions}
		model = {}
		policy = {state: {action: 1.0/len(actions) for action in actions} for state in states}
		steps = []
		rewards = []
		cum_reward = 0
		queue = PriorityQueue()
		for i in range(1):
			num_steps = 0
			generator = generator_class(*generator_args)
			current_state = generator.state
			#while True:
			for k in range(2000):
				num_steps += 1
				action, next_state, reward = generator.step(policy, current_state)
				cum_reward += reward
				rewards.append(cum_reward)
				if next_state == None:
					action_values[(current_state,action)] += alpha*(reward-action_values[(current_state,action)])
				else:
					max_value = max(action_values[(next_state,a)] for a in actions)
					possible_actions = [a for a in actions if action_values[(next_state,a)] > (max_value-0.01)]
					next_action = possible_actions[np.random.randint(len(possible_actions))]
					action_values[(current_state,action)] += alpha*(reward+gamma*action_values[(next_state,next_action)]-action_values[(current_state,action)])
				model[(current_state,action)] = (reward,next_state)
				current_state = next_state
				if next_state == None:
					break
				for j in range(n):
					action_state = model.keys()[np.random.randint(len(model.keys()))]
					reward, next_state = model[action_state]
					if next_state != None:
						next_action = max(policy[next_state],key=policy[next_state].get)
						action_values[action_state] += alpha*(reward+gamma*action_values[(next_state,next_action)]-action_values[action_state])
				# update policy
				for state in policy:
					best_action = None
					best_action_value = -float('inf')
					for action in actions:
						if action_values[(state,action)] > best_action_value:
							best_action = action
							best_action_value = action_values[(state,action)]
					if best_action_value > 0:
						policy[state] = {action: epsilon/len(actions) for action in actions}
						policy[state][best_action] += 1-epsilon
					else:
						policy[state] = {action: 1.0/len(actions) for action in actions}
			steps.append(num_steps)
		# episode_steps = 1.0*(episode_steps*trial+np.array(steps))/(trial+1)
		episode_rewards = 1.0*(episode_rewards*trial+np.array(rewards))/(trial+1)
	
	#return action_values, policy, episode_steps
	return action_values, policy, episode_rewards

# reproduce Figure 8.5
def maze():
	states = [(i,j) for i in range(9) for j in range(6)]
	actions = [(-1,0),(1,0),(0,-1),(0,1)]
	action_values, policy, num_steps = dyna_q(states, actions, MazeGenerator, [], 50)
	plt.plot(range(5), num_steps)
	plt.show()

def blocking():
	states = [(i,j) for i in range(9) for j in range(6)]
	actions = [(-1,0),(1,0),(0,-1),(0,1)]
	action_values, policy, rewards = dyna_q(states, actions, BlockingGenerator, [], 5)
	action_values, policy, rewards2 = dyna_q_plus(states, actions, BlockingGenerator, [], 5)
	plt.plot(range(2000), rewards, 'r', range(2000), rewards2, 'b')
	plt.show()

blocking()
