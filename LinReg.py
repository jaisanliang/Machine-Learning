'''
module for doing gradient descent to find best parameters
user provides data in the form of a csv file
each line is a feature vector
'''

import pandas
import csv
import re
import Auxiliary

numFeatures = Test.getNumFeatures()
learnRate = 0.001

featureWeights = [0 for i in range(numFeatures)]

def getFeatures():
    return

def calculateWeights(vectors):
    weights = [0 for i in range(numFeatures)]
    prevWeights = weights
    while not Auxiliary.converged(weights,prevWeights):
        prevWeights = weights
        weights = batchDescentStep(vectors,weights)
    return weights

def batchDescentStep(vectors,weights):
    for i in range(numFeatures):
        derivative = sum(vectorCost(vector,weights) for vector in vectors)
        weights[i] -= learnRate*derivative
    return weights

def vectorCost(vector,weights):
    predictedVal = sum(vectors[i]*weights[i] for i in range)
    return vectors[0]-predictedVal

def stochasticDescentStep(vectors,weights):
    return vectorCost(vector,weights)
