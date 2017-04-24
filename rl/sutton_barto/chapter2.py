import pdb

import math
import numpy as np
import matplotlib.pyplot as plt

timesteps = 1000
tasks = 10
alpha = 0.1

def epsilon_greedy_selection(e_action_values, n, epsilon, temperature):
	return np.where(np.random.rand(tasks) < epsilon, np.random.randint(n, size=tasks), np.argmax(e_action_values, axis=1))

def gibbs_selection(e_action_values, n, epsilon, temperature):
	return np.apply_along_axis(lambda x: np.random.choice(n, p=math.e**((x-np.max(x))/temperature)/(np.sum(math.e**((x-np.max(x))/temperature)))), axis=1, arr=e_action_values).astype(int)

def solve_bandits(n, epsilon=0, initial_value=0, action_selection = epsilon_greedy_selection, temperature = 1, average_update = True):
	action_values = np.random.normal(size=(tasks,n))
	optimal_actions = np.argmax(action_values, axis=1)
	e_action_values = initial_value*np.ones((tasks,n))
	n_action_sampled = np.zeros((tasks,n))
	frac_optimal = np.zeros((tasks,timesteps))
	rewards = np.zeros((tasks,timesteps))
	for i in range(timesteps):
		# make the values take random walks for nonstationary problems
		# if i % 100 == 0:
		#	action_values = np.random.normal(size=(tasks,n))
		#	optimal_actions = np.argmax(action_values, axis=1)
		actions = action_selection(e_action_values, n, epsilon, temperature)
		reward = action_values[range(tasks),actions] + np.random.normal(size=tasks)
		# sample average update
		if average_update:
			e_action_values[range(tasks),actions] += (reward-e_action_values[range(tasks),actions])/(n_action_sampled[range(tasks),actions]+1)
		# constant value update
		else:
			e_action_values[range(tasks),actions] += alpha*(reward-e_action_values[range(tasks),actions])
		n_action_sampled[range(tasks),actions] += 1
		frac_optimal[:,i] = np.equal(actions,optimal_actions).astype(int)
		rewards[:,i] = reward
	return np.mean(rewards,axis=0), np.mean(frac_optimal,axis=0)

# reproducing Figure 2.1
'''
rewards_0, frac_optimal_0 = solve_bandits(10)
rewards_01, frac_optimal_01 = solve_bandits(10, epsilon=0.1)
rewards_001, frac_optimal_001 = solve_bandits(10, epsilon=0.01)
plt.plot(range(timesteps),rewards_0,'g',rewards_01,'k',rewards_001,'r')
plt.show()
plt.plot(range(timesteps),frac_optimal_0,'g',frac_optimal_01,'k',frac_optimal_001,'r')
plt.show()
'''

# Exercise 2.2
'''
rewards_g, frac_optimal_g = solve_bandits(10)
rewards_01, frac_optimal_01 = solve_bandits(10, action_selection=gibbs_selection, temperature=0.000001)
rewards_1, frac_optimal_1 = solve_bandits(10, action_selection=gibbs_selection, temperature=1)
rewards_10, frac_optimal_10 = solve_bandits(10, action_selection=gibbs_selection, temperature=1000000)
plt.plot(range(timesteps),rewards_g,'g',rewards_01,'k',rewards_1,'r',rewards_10,'b')
plt.show()
plt.plot(range(timesteps),frac_optimal_g,'g',frac_optimal_01,'k',frac_optimal_1,'r',frac_optimal_10,'b')
plt.show()
'''

# Exercise 2.6
'''
rewards_a, frac_optimal_a = solve_bandits(10, epsilon=0.1)
rewards_c, frac_optimal_c = solve_bandits(10, epsilon=0.1, average_update = False)
plt.plot(range(timesteps),rewards_a,'g',rewards_c,'k')
plt.show()
plt.plot(range(timesteps),frac_optimal_a,'g',frac_optimal_c,'k')
plt.show()
'''

# reproducing Figure 2.2
'''
rewards_o, frac_optimal_o = solve_bandits(10, epsilon=0, initial_value=5, average_update = False)
rewards_g, frac_optimal_g = solve_bandits(10, epsilon=0.1, average_update = False)
plt.plot(range(timesteps),frac_optimal_o,'g',frac_optimal_g,'k')
plt.show()
'''
