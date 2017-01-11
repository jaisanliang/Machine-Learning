'''
Module that contains auxiliary functions for machine learning
'''

import numpy as np

class Classifier:
    def train(self, x, y):
        assert len(x) == len(y)
        assert len(x) > 0
    def classify(self, x):
        pass
    def test(self, x, y):
        pass

def normalize(vectors):
    '''
    Normalize the vectors by expressing each vector in standard normal form
    '''
    normVectors=np.float_(vectors)
    # get avg and std of each feature
    means=np.average(normVectors,axis=0)
    stds=np.std(normVectors,axis=0)
    for i in range(len(normVectors)):
        for j in range(len(normVectors[i])):
            # take care of the case where std=0
            if stds[j]==0:
                normVectors[i][j]=0.0
            else:
                normVectors[i][j]=(1.0*normVectors[i][j]-means[j])/stds[j]
    return normVectors
        
def has_converged(weights,prevWeights):
    '''
    Determine if the learned weights for a feature have converged
    '''
    numFeatures=len(weights)
    epsilon=0.0001
    for i in range(numFeatures):
        if (weights[i]-prevWeights[i])>epsilon:
            return False
    return True

def kmp(string, substring):
    '''
    Return list of indices i of string such that string[i:i+len(substring)] == substring
    https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm
    '''

def rabin_karp(string, substring):
    '''
    Return list of indices i of string such that string[i:i+len(substring)] == substring
    https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm
    '''
    pass

def regularize(matrix, l):
    n = len(matrix)
    return np.add(matrix, l*np.identity(n))

def largest_off_diagonal(matrix):
    n = len(matrix)
    inf_diagonal = -float('inf')*np.identity(n)
    matrix = np.add(matrix, inf_diagonal)
    return np.amax(matrix)



#def precision(actualVsPredict):
    

#def recall(actualVsPredict):
