import pdb

import math
import numpy as np
import matplotlib.pyplot as plt

# DP implementation of greedy policy iteration
def policy_iteration(states, transitions, rewards, gamma):
	'''
	transitions: dict[state][action][state'] with transition probabilities
	rewards: dict[state][action][state'] with average rewards for each transition
	'''
	# initialize random policy, state values
	values = {state: 0 for state in states}
	policies = {state: transitions[state].keys()[0] for state in transitions}
	# loop until policy is unchanged
	while True:
		# policy evaluation
		# loop until state values converge
		while True:
			delta = 0
			newvalues = {state: 0 for state in states}
			for state in transitions:
				action = policies[state]
				for next_state, prob in transitions[state][action].iteritems():
					newvalues[state] += prob*(rewards[state][action][next_state]+gamma*values[next_state])
			for state in values:
				delta = max(delta,abs(values[state]-newvalues[state]))
			if delta < 1e-6:
				break
			values = newvalues
		# policy improvement
		policy_stable = True
		newpolicies = {state: policies[state] for state in policies}
		for state in newpolicies:
			best_action = None
			best_value = -float('inf')
			for action in transitions[state]:
				value = 0
				for next_state, prob in transitions[state][action].iteritems():
					value += prob*(rewards[state][action][next_state]+gamma*values[next_state])
				if value > best_value:
					best_value = value
					best_action = action
			newpolicies[state] = best_action
			if policies[state] != newpolicies[state]:
				policy_stable = False
		# plt.plot(values.keys(), values.values())
		# plt.show()
		policies = newpolicies
		if policy_stable:
			# plt.plot(policies.keys(), policies.values())
			# plt.show()
			return
		
# DP implementation of greedy value iteration
def value_iteration(states, transitions, rewards, gamma):
	'''
	transitions: dict[state][action][state'] with transition probabilities
	rewards: dict[state][action][state'] with average rewards for each transition
	'''
	# initialize random policy, state values
	# terminal states have value of 0 forever
	values = {state: 0 for state in states}
	policies = {state: transitions[state].keys()[0] for state in transitions}
	# loop until values have converged
	while True:
		# policy evaluation
		delta = 0
		# use transitions instead of states so that terminal states are not updated
		for state in transitions:
			best_action = None
			best_value = -float('inf')
			# greedily select an action
			for action in transitions[state]:
				value = 0
				for next_state, prob in transitions[state][action].iteritems():
					value += prob*(rewards[state][action][next_state]+gamma*values[next_state])
				if value > best_value:
					best_value = value
					best_action = action
			delta = max(delta,abs(values[state]-best_value))
			values[state] = best_value
			policies[state] = best_action
		# print policies
		# plt.plot(policies.keys(), policies.values())
		# plt.plot(values.keys(), values.values())
		# plt.show()
		if delta < 1e-14:
			plt.plot(policies.keys(), policies.values())
			plt.show()
			return

def poisson_helper(l, n):
	return l**n*math.e**(-l)/math.factorial(n)

'''
poisson_array = np.zeros([5,10])
for i in range(1,5):
	for j in range(10):
		poisson_array[i][j] = poisson_helper(i,j)
'''

def poisson(l, n):
	return poisson_array[l,n]

def car_rental():
	'''
	states: the number of cars in each lot at the end of each day
	actions: number of cars moved from lot 1 to lot 2 during the night
	'''
	max_cars = 20
	gamma = 0.9
	# parameters for poisson distribution
	request1 = 3
	request2 = 4
	return1 = 3
	return2 = 2
	actions = range(-5,6)
	states = [(i,j) for i in range(max_cars+1) for j in range(max_cars+1)]
	transitions = {}
	rewards = {}
	for state in states:
		transitions[state] = {}
		rewards[state] = {}
		for action in actions:
			# can't move cars so that there are more than 20 or less than 0 cars in a lot
			if (state[0] - action <= max_cars) and (state[1] + action <= max_cars) and (state[0] - action >= 0) and (state[1] + action >= 0):
				transitions[state][action] = {}
				rewards[state][action] = {}
				# baseline for next state is [state[0]-action,state[1]+action]
				# next state is a function of the number of cars requested and the number of cars returned
				# truncate unlikely request/return numbers to make computation faster
				for requests1 in range(max_cars/2):
					for requests2 in range(max_cars/2):
						for returns1 in range(max_cars/2):
							for returns2 in range(max_cars/2):
								# calculate number of cars in each lot after requests and returns
								newstate1 = min(max(state[0]-action-requests1,0)+returns1,max_cars)
								newstate2 = min(max(state[1]+action-requests2,0)+returns2,max_cars)
								newstate = (newstate1,newstate2)
								if newstate not in transitions[state][action]:
									transitions[state][action][newstate] = 0
									rewards[state][action][newstate] = 0
								prob = poisson(request1,requests1)*poisson(request2,requests2)*poisson(return1,returns1)*poisson(return2,returns2)
								transitions[state][action][newstate] += prob
				for requests1 in range(max_cars/2):
					for requests2 in range(max_cars/2):
						for returns1 in range(max_cars/2):
							for returns2 in range(max_cars/2):
								newstate1 = min(max(state[0]-action-requests1,0)+returns1,max_cars)
								newstate2 = min(max(state[1]+action-requests2,0)+returns2,max_cars)
								newstate = (newstate1,newstate2)
								prob = poisson(request1,requests1)*poisson(request2,requests2)*poisson(return1,returns1)*poisson(return2,returns2)
								rewards[state][action][newstate] += \
									prob/transitions[state][action][newstate]*(10*(min(state[0]-action,requests1)+min(state[1]+action,requests2))-2*abs(action))
	policy_iteration(states, transitions, rewards, gamma)

def gambler(p=0.5):
	gamma = 1
	states = range(101)
	transitions = {}
	rewards = {}
	for state in states:
		transitions[state] = {}
		rewards[state] = {}
		for action in range(min(state,100-state)+1):
			if action == 0:
				transitions[state][action] = {state: 1}
			else:
				transitions[state][action] = {(state+action): p, (state-action): (1-p)}
			rewards[state][action] = {(state+action): int(state+action == 100), (state-action): 0}
	# delete terminal state transitions and rewards
	del transitions[100]
	del rewards[100]
	value_iteration(states, transitions, rewards, gamma)
