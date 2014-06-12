'''
Module that handles background work such as dividing
data into training, cross-validation, and test sets
'''

import pandas
import csv
import re
import random

# number of folds to use in doing cross-validation

def getFeatures(fileName,numFolds):
    '''
    Returns a Dataframe with training and test data
    First column is the correct value, the other
    columns are feature values
    Also divides data into folds
    '''
    vectors = pandas.DataFrame.from_csv(fileName)
    vectors['Fold Number'] = random.randit(1,numFolds)
    vectors.groupby('Fold Number')
    return vectors

def train(vectors,foldNum):
    

def getCrossValidation(vectors,foldNum):
    

def normalize:
    featureMeans = featureWeights.mean()
    featureStds = featureWeights.std()
    
    for index, row in featureVectors.iterrows():
        
def converged(weights,prevWeights):
    epsilon = 0.0001
    for i in range(numFeatures):
        if (weights[i]-prevWeights[i]) > epsilon:
            return False
    return True
