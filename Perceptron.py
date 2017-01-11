from Auxiliary import *
import numpy as np

class Perceptron(Classifier):
	'''
	Perceptron algorithm
	Takes in feature vectors of dimension d, where each vector is in class 0 or 1
	Finds best separating hyperplane through origin given training set
	'''
	def __init__(self):
		self.w = None

	def train(self,X,Y):
		d = len(X[0])
		n = len(Y)
		# augment training vectors with Theta_0=1
		X_aug = np.hstack((np.ones((n,1)), X))
		self.w = np.zeros(d+1)
		for i in range(n):
			sign = (np.dot(self.w, X_aug[i]) > 0)
			self.w += (Y[i]-sign)*X_aug[i]
	def test(self, X):
		return [np.sign(np.dot(self.w), x) for x in X]

classifier = Perceptron()
classifier.train([[1,1],[-1,-1]],[1,0])
print classifier.w