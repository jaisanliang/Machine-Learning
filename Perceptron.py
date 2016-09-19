from Auxiliary import *
import numpy as np

class Perceptron(Classifier):
	def __init__(self):
		self.v_normal = None

	def train(self, x, y):
		Classifier.train(self, x, y)
		d = len(x[0])
		n = len(y)
		# augment training vectors with Theta_0=1
		x_aug = np.hstack((np.ones((n,1)), x))
		v_normal = np.zeros(d + 1)
		for i in range(n):
			sign = np.sign(np.dot(v_normal, x_aug[i]))
			if sign <= 0:
				v_normal = np.add(v_normal, sign*x_aug[i])
		self.v_normal = v_normal
	def test(self, x):
		Classifier.test(self, x)
		return [np.sign(np.dot(self.v_normal), x_i) for x_i in x]

classifier = Perceptron()
classifier.train([[0,0]],[1])