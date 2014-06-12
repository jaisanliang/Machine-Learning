'''
Module for doing classification using
a logistic model to find feature weights
'''

import math
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
        weights = descentStep(vectors,weights)
    return weights

def vectorCost(vector,weights):
    predictedVal = sum(vector[i]*weights[i] for i in range(numFeatures))
    return vector[numFeatures]-1.0/(1+math.e**(-predictedVal))

def descentStep(vectors,weights):
    newWeights = weights
    for i in range(numFeatures):
        derivative = sum(vectorCost(vectors.loc[j],weights)*vectors.loc[j][i] for j in range(len(vectors)))
        newWeights[i] += learnRate*derivative
    return newWeights
