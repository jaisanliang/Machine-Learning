"""
Implementation of K-means algorithm
"""

import Auxiliary
import numpy as np
import pandas as pd
import random

# number of centers
K = 2

def distance(vector1,vector2):
    """Calculates Cartesian distance between two feature vectors"""
    return sum((vector1[i]-vector2[i])^2 for i in range(len(vector1)))^0.5

def find_closest(vector,centers):
    """Finds index of closest center to vector"""
    closestDist = distance(vector, centers[0])
    closestCenter = 0
    for i in len(centers):
        center = centers[i]
        if distance(vector,center) < closestDist:
            closestDist = distance(vector,center)
            closestCenter = i
    return closestCenter

def all_converged(centers,lastCenters):
    """Checks if centers have converged to a stable equilibrium"""
    for i in range(len(centers)):
        if not Auxiliary.converged(centers[i],lastCenters[i]):
            return False
    return True

def get_centers(vectors):
    """Main method for finding cluster centers"""
    numFeatures = len(vectors[0])
    # randomly choose k starting vectors
    centers = np.array([])
    # copy vectors before choosing centers so that centers aren't repeated
    newVectors = vectors
    indexedVectors = pd.DataFrame(vectors)
    closestIndex = [0 for i in range(len(vectors))]
    for i in range(K):
        newStart = newVectors[random.randint(0,len(vectors)-1)]
        newVectors = np.delete(newVectors,newStart)
        centers.append(newStart)
    # repeat algorithm until convergence
    while True:
        lastCenters = centers
        for i in len(vectors):
            closestCenter = findClosest(vector,centers)
            closestIndex[i] = closestCenter
        indexedVectors['Closest Center'] = closestIndex
        groupedVectors = indexedVectors.groupby('Closest Center')
        centers = np.array(grouped.mean())
        if all_converged(centers,lastCenters):
            break
    return centers
