"""
Implementation of K-means algorithm
"""

import pandas
import random

k = 2

def distance(vector1,vector2):
    return sum((vector1[i]-vector2[i])^2 for i in range(len(vector1)))

def findClosest(vector,centers):
    closestDist = distance(vector, centers[0])
    closestCenter = 0
    for i in len(centers):
        center = centers[i]
        if distance(vector,center) < closestDist:
            closestDist = distance(vector,center)
            closestCenter = i
    return closestCenter

def allConverged(centers,lastCenters):
    for i in range(len(centers)):
        if not Auxiliary.converged(centers[i],lastCenters[i]):
            return False
    return True

def getClusterCenters(vectors):
    numFeatures = len(vectors.loc[0])
    # randomly choose k starting vectors
    centers = pandas.Series([])
    for i in range(k):
        newStart = vectors.loc[random.randint(1,len(vectors))]
        centers.append(pandas.Series([newStart]))
    lastCenters = centers
    # repeat algorithm until convergence
    while not allConverged(centers,lastCenters):
        lastCenters = centers
        for i in len(vectors):
            closestCenter = findClosest(vector,centers)
            vectors['Closest Center'] = closestCenter
        grouped = vectors.groupby('Closest Center')
        centers = grouped.mean()
    return centers
