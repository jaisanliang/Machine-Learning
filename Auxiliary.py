'''
Module that contains auxiliary functions for machine learning,
including calculators for precision and recall
'''

import numpy as np

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

#def precision(actualVsPredict):
    

#def recall(actualVsPredict):
