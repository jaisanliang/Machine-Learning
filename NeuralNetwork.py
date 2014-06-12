'''
Module for neural network
'''

import math
import random
import pandas
import Auxiliary

# each element of the array is
layers = []
numLayers = len(layers)
numFeatures = 0
initRange = 0.01
learnRate = 0.001

def calculateWeights(vectors):
    '''
    vector is DataFrame with correct value at end
    '''
    weights = pandas.Series([[random.uniform(-initRange,initRange) for i in range(layers[j])] for j in range(numLayers)])
    prevWeights = weights
    while not Auxiliary.converged(weights,prevWeights):
        prevWeights = weights
        weights = descentStep(vectors,weights)
    return weights

def forwardPropagate():
    

def backPropagate():
    

def vectorCost(vector,weights):
    predictedVal = sum(vector[i]*weights[i] for i in range(numFeatures))
    return vector[numFeatures]-1.0/(1+math.e**(-predictedVal))

def descentStep(vectors,weights):
    newWeights = weights
    for i in range(numFeatures):
        derivative = sum(vectorCost(vectors.loc[j],weights)*vectors.loc[j][i] for j in range(len(vectors)))
        newWeights[i] += learnRate*derivative
    return newWeights
