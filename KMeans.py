"""
Implementation of K-means algorithm
Assumes user inputs data in the form of a list of feature vectors
"""

import pandas
import csv
import re
import random

def distance(vector1,vector2):
    return sum((vector1[i]-vector2[i])^2 for i in range(len(vector1)))

def findClosest(vector,meansList):
    closestDist = distance(vector, meansList[0])
    closestMean = meansList[0]
    for i in len(meansList):
        mean = meansList[i]
        if distance(vector,mean) < closestDist:
            closestDist = distance(vector,mean)
            closestMean = i

inputFilePath = input('Path to input file: ')
vectorList = DataFrame.from_csv(inputFilePath)
#get the number of features in each vector
numFeatures = len(vectorList[0])

#set K
k = 2

# randomly choose k starting vectors, pop them so we don't get duplicates
meansList = []
for i in range(k):
    newStart = vectorList.pop(random.choice(vectorList))
    startVectors.append(newStart)
for i in range(k):
    vectorList.append(startVectors[i])

# repeat algorithm until convergence
for i in range(10):
    closestAssignments = dict([] for j in range(k))
    for vector in vectorList:
        index = findClosest(vector,meansList)
        closestAssignments[index].append
