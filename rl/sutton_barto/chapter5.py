import pdb
import sys

import math
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import pylab

class BlackjackGenerator():
	def __init__(self, start_state = None):
		# draw two cards each for player and dealer, have dealer show one card randomly
		if start_state == None:
			player_cards = [min(np.random.randint(1,13),10) for i in range(2)]
			self.dealer_cards = [min(np.random.randint(1,13),10) for i in range(2)]
			# return initial state
			player_sum = sum(player_cards)
			if player_cards[0] == 1 or player_cards[1] == 1:
				useable_ace = 1
				player_sum += 10
			else:
				useable_ace = 0
			# TODO: check if player has natural
			# if player_sum == 21:
			#	self.state = None
			self.state = (useable_ace, self.dealer_cards[0], player_sum)
		else:
			self.dealer_cards = [start_state[1],min(np.random.randint(1,13),10)]
			self.state = start_state
	def step(self, policy, current_state, action = None):
		useable_ace, dealer_showing, player_sum = current_state
		# 0 for stick, 1 for hit
		if action == None:
			action = np.random.choice(policy[current_state].keys(), p=policy[current_state].values())
		if action == 0:
			# player sticks, dealer draws if his sum < 17 (use as many aces as possible), and sticks otherwise
			dealer_sum = sum(self.dealer_cards)
			while dealer_sum < 17:
				if 1 in self.dealer_cards and dealer_sum < 12:
					self.dealer_cards.remove(1)
					self.dealer_cards.append(11)
					dealer_sum += 10
				card = min(np.random.randint(1,13),10)
				# see if the dealer has gone bust
				if dealer_sum + card > 21:
					if 11 not in self.dealer_cards:
						return (action,None,1)
					else:
						self.dealer_cards.remove(11)
						self.dealer_cards.append(1)
				self.dealer_cards.append(card)
				dealer_sum = sum(self.dealer_cards)
			if dealer_sum > player_sum:
				return (action,None,-1)
			elif dealer_sum == player_sum:
				return (action,None,0)
			else:
				return (action,None,1)
		else:
			# give the player a random card, all face cards count as 10
			card = min(np.random.randint(1,13),10)
			player_sum += card
			# check if player goes bust
			if player_sum > 21:
				if useable_ace == 0:
					return (action,None,-1)
				else:
					player_sum -= 10
					useable_ace = 0
					return (action,(useable_ace,dealer_showing,player_sum),0)
			else:
				if card == 1 and player_sum+10 < 22:
					useable_ace = 1
					player_sum += 10
				return (action,(useable_ace,dealer_showing,player_sum),0)

# first-visit Monte Carlo
def monte_carlo_fv(states, generator_class):
	gamma = 1
	# initialize rewards for each state, random policy
	values = {state: 0 for state in states}
	returns = {state: [] for state in states}
	# for blackjack, test the policy which sticks on only 20 and 21
	policies = {state: {0: 0, 1: 1} if (state[2] != 20 and state[2] != 21) else {0: 1, 1: 0} for state in states}
	for i in range(100000):
		if i % 10000 == 0:
			print i
		generator = generator_class()
		current_state = generator.state
		episode_states = []
		episode_rewards = []
		# generate random episode using policy
		while True:
			_, next_state, reward = generator.step(policies,current_state)
			episode_states.append(current_state)
			episode_rewards.append(reward)
			if next_state == None:
				break
			current_state = next_state
		# find returns following first occurrence of each state
		episode_returns = [0 for i in range(len(episode_states))]
		episode_returns[-1] = episode_rewards[-1]
		for i in range(len(episode_states)-2,-1,-1):
			episode_returns[i] = episode_rewards[i]+gamma*episode_returns[i+1]
		seen_states = set()
		for i, state in enumerate(episode_states):
			if state not in seen_states:
				seen_states.add(state)
				returns[state].append(episode_returns[i])
				values[state] = np.average(returns[state])
	return values

# Monte Carlo with exploring starts
def monte_carlo_es(states, actions, generator_class):
	gamma = 1
	action_values = {(state,action): 0 for state in states for action in actions}
	# for blackjack, start with the policy which sticks on only 20 and 21
	policies = {state: {0: 0, 1: 1} if (state[2] != 20 and state[2] != 21) else {0: 1, 1: 0} for state in states}
	returns = {(state,action): [] for state in states for action in actions}
	for i in range(500000):
		# initialize with random state, generate rest of episode
		current_state = states[np.random.randint(len(states))]
		generator = generator_class(current_state)
		current_action = actions[np.random.randint(len(actions))]
		current_action_state = (current_state,current_action)
		episode_action_states = []
		episode_rewards = []
		# generate random episode using policy and starting with random state and action
		while True:
			action, next_state, reward = generator.step(policies,current_state,current_action)
			episode_action_states.append((current_state,action))
			episode_rewards.append(reward)
			if next_state == None:
				break
			current_state = next_state
			current_action = None
		# find returns following first occurrence of each state
		episode_returns = [0 for i in range(len(episode_action_states))]
		episode_returns[-1] = episode_rewards[-1]
		for i in range(len(episode_action_states)-2,-1,-1):
			episode_returns[i] = episode_rewards[i]+gamma*episode_returns[i+1]
		seen_action_states = set()
		for i, action_state in enumerate(episode_action_states):
			if action_state not in seen_action_states:
				seen_action_states.add(action_state)
				returns[action_state].append(episode_returns[i])
				action_values[action_state] = np.average(returns[action_state])
		# update policy
		for state in policies:
			best_action = None
			best_action_value = -float('inf')
			for action in actions:
				policies[state][action] = 0
				if action_values[(state,action)] > best_action_value:
					best_action = action
					best_action_value = action_values[(state,action)]
			policies[state][best_action] = 1
	return policies, action_values

# e-soft on-policy Monte Carlo control
def monte_carlo_on_p(states, actions, generator_class):
	gamma = 1
	epsilon = 0.1
	action_values = {(state,action): 0 for state in states for action in actions}
	# policies = {state: {action: 1.0/len(actions) for action in actions} for state in states}
	policies = {state: {0: 0.5, 1: 0.5} if state[2] > 11 else {0: 0, 1: 1} for state in states}
	returns = {(state,action): [] for state in states for action in actions}
	for i in range(500000):
		# initialize with start state, generate rest of episode
		generator = generator_class()
		current_state = generator.state
		episode_action_states = []
		episode_rewards = []
		# generate random episode using policy
		while True:
			action, next_state, reward = generator.step(policies,current_state)
			episode_action_states.append((current_state,action))
			episode_rewards.append(reward)
			if next_state == None:
				break
			current_state = next_state
			current_action = None
		# find returns following first occurrence of each state
		episode_returns = [0 for i in range(len(episode_action_states))]
		episode_returns[-1] = episode_rewards[-1]
		for i in range(len(episode_action_states)-2,-1,-1):
			episode_returns[i] = episode_rewards[i]+gamma*episode_returns[i+1]
		seen_action_states = set()
		for i, action_state in enumerate(episode_action_states):
			if action_state not in seen_action_states:
				seen_action_states.add(action_state)
				returns[action_state].append(episode_returns[i])
				action_values[action_state] = np.average(returns[action_state])
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
	return policies, action_values

# off-policy Monte Carlo control
def monte_carlo_off_p(states, actions, generator_class):
	gamma = 1
	epsilon = 0.1
	action_values = {(state,action): 0 for state in states for action in actions}
	action_values_n = {(state,action): 0 for state in states for action in actions}
	action_values_d = {(state,action): 0 for state in states for action in actions}
	# for blackjack, start with the policy which sticks on only 20 and 21
	policies = {state: {0: 0, 1: 1} if (state[2] != 20 and state[2] != 21) else {0: 1, 1: 0} for state in states}
	returns = {(state,action): [] for state in states for action in actions}
	for i in range(500000):
		# make e-greedy off-policy
		off_policy = {}
		for state in states:
			off_policy[state] = {}
			for action in actions:
				if policies[state][action] == 1:
					off_policy[state][action] = 1-epsilon+epsilon/len(actions)
				else:
					off_policy[state][action] = epsilon/len(actions)
		# generate rest of episode
		generator = generator_class()
		current_state = generator.state
		episode_action_states = []
		episode_rewards = []
		# generate random episode using policy and starting with random state and action
		while True:
			action, next_state, reward = generator.step(off_policy,current_state)
			episode_action_states.append((current_state,action))
			episode_rewards.append(reward)
			if next_state == None:
				break
			current_state = next_state
		# find returns following first occurrence of each state and probability ratios for each ending sequence
		episode_returns = [0 for i in range(len(episode_action_states))]
		w = [0 for i in range(len(episode_action_states))]
		episode_returns[-1] = episode_rewards[-1]
		w[-1] = 1
		for i in range(len(episode_action_states)-2,-1,-1):
			episode_returns[i] = episode_rewards[i]+gamma*episode_returns[i+1]
			next_state,next_action = episode_action_states[i+1]
			w[i] = w[i+1]/off_policy[next_state][next_action]
		seen_action_states = set()
		# find latest time at which action is not greedy action
		t = 0
		for i in range(len(episode_action_states)-1,-1,-1):
			state, action = episode_action_states[i]
			is_greedy_action = policies[state][action] == 1
			if not is_greedy_action:
				t = i
				break
		for i, action_state in enumerate(episode_action_states):
			if i >= t:
				if action_state not in seen_action_states:
					seen_action_states.add(action_state)
					action_values_n[action_state] += w[i]*episode_returns[i]
					action_values_d[action_state] += w[i]
					action_values[action_state] = action_values_n[action_state]/action_values_d[action_state]
		# update policy
		for state in policies:
			best_action = None
			best_action_value = -float('inf')
			for action in actions:
				policies[state][action] = 0
				if action_values[(state,action)] > best_action_value:
					best_action = action
					best_action_value = action_values[(state,action)]
			policies[state][best_action] = 1
	return policies, action_values

def blackjack():
	blackjack_states = []
	for useable_ace in range(2):
		for dealer_showing in range(1,11):
			for player_sum in range(4,22):
				blackjack_states.append((useable_ace,dealer_showing,player_sum))
	# values = monte_carlo_fv(blackjack_states,BlackjackGenerator)
	# replicate Figure 5.2
	'''
	fig = pylab.figure()
	ax = Axes3D(fig)
	x = []
	for i in range(12,22):
		for j in range(10):
			x.append(i)
	ax.scatter(x, range(1,11)*10, [values[0,i,j] for j in range(12,22) for i in range(1,11)])
	plt.show()
	'''

	policies, action_values = monte_carlo_off_p(blackjack_states,[0,1],BlackjackGenerator)
	# replicate Figure 5.5
	for player_sum in range(21,11,-1):
		for dealer_showing in range(1,11):
			sys.stdout.write(str(int(policies[(0,dealer_showing,player_sum)][0]<0.5))+' ')
		sys.stdout.write('\n')
	print '\n'
	for player_sum in range(21,11,-1):
		for dealer_showing in range(1,11):
			sys.stdout.write(str(int(policies[(1,dealer_showing,player_sum)][0]<0.5))+' ')
		sys.stdout.write('\n')
