'''
Module for doing linear regression using both
batch and stochastic gradient descent to find feature weights
'''

import pandas
import Auxiliary

numFeatures = 0
learnRate = 0.001

def calculateWeights(vectors):
    '''
    vector is DataFrame with correct value at end
    '''
    numFeatures = len(vectors.loc[0])-1
    weights = pandas.Series([0 for i in range(numFeatures)])
    prevWeights = weights
    while not Auxiliary.converged(weights,prevWeights):
        prevWeights = weights
        weights = batchDescentStep(vectors,weights)
    return weights

def vectorCost(vector,weights):
    predictedVal = sum(vector[i]*weights[i] for i in range(numFeatures))
    return vector[numFeatures]-predictedVal

def batchDescentStep(vectors,weights):
    newWeights = weights
    for i in range(numFeatures):
        derivative = sum(vectorCost(vectors.loc[j],weights)*vectors.loc[j][i] for j in range(len(vectors)))
        newWeights[i] += learnRate*derivative
    return newWeights

def stochasticDescentStep(vector,weights):
    newWeights = weights
    for i in range(numFeatures):
        derivative = vectorCost(vector,weights)*vector[i]
        newWeights[i] += learnRate*derivative
    return newWeights
