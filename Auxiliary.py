'''
Module that handles background work such as dividing
data into training, cross-validation, and test sets
'''

import pandas
import csv
import re

def normalize(vectors):
    featureMeans = featureWeights.mean()
    featureStds = featureWeights.std()
    
    for index, row in featureVectors.iterrows():
        
def converged(weights,prevWeights):
    epsilon = 0.0001
    for i in range(numFeatures):
        if (weights[i]-prevWeights[i]) > epsilon:
            return False
    return True
    
