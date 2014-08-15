'''
Module for neural network
'''

import math
import random
import numpy as np
import pandas as pd
import Auxiliary

# each element of the layers array is the number of units in that network
LAYERS = []
NUMLAYERS = len(layers)
NUMFEATURES = 0
INITRANGE = 0.01
LEARNRATE = 0.001

def vector_cost(vector,weights):
    """Calculates difference between actual and predicted (given feature vector) value"""
    predictedVal = sum(vector[i]*weights[i] for i in range(NUMFEATURES))
    return vector[NUMFEATURES]-1.0/(1+math.e**(-predictedVal))

def batchDescentStep(vectors,weights):
    """Calculates new feature weights after one batch descent step"""
    newWeights = weights
    for i in range(NUMFEATURES):
        derivative = sum(vector_cost(vectors[j],weights)*vectors[j][i] for j in range(len(vectors)))
        newWeights[i] += LEARNRATE*derivative
    return newWeights

def stochasticDescentStep(vector,weights):
    """Calculates new feature weights after one stochastic descent step"""
    newWeights = weights
    for i in range(NUMFEATURES):
        derivative = vector_cost(vector,weights)*vector[i]
        newWeights[i] += LEARNRATE*derivative
    return newWeights

def calculate_weights(vectors,batch=True):
    """
    Main method for calculating weights of features
    Vector is feature vector with correct classification appended
    """
    NUMFEATURES = len(vectors[0])-1
    weights = np.array([[random.uniform(-INITRANGE,INITRANGE) for i in range(LAYERS[j])] for j in range(NUMLAYERS)])
    stochasticIndex = 0
    while True:
        prevWeights = weights
        if batch:
            weights = batchDescentStep(vectors,weights)
        else:
            weights = stochasticDescentStep(vectors[stochasticIndex],weights)
            stochasticIndex = (stochasticIndex+1)%len(vectors)
        if Auxiliary.converged(weights,prevWeights):
            return weights

def forwardPropagate():
    

def backPropagate():
    
