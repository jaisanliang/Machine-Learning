'''
Module that contains auxiliary functions,
including calculators for precision and recall
'''

import pandas

def normalize(vectors):
    means = vectors.mean()
    stds = vectors.std()
    for i in range(len(vectors)):
        for j in range(len(vectors.loc[i])):
            # what is std=0?
            vectors.loc[i][j] = (vectors.loc[i][j]-means[j])/stds[j]
    return vectors
        
def converged(weights,prevWeights):
    epsilon = 0.0001
    for i in range(numFeatures):
        if (weights[i]-prevWeights[i]) > epsilon:
            return False
    return True

def precision(actualVsPredict):
    

def recall(actualVsPredict):
