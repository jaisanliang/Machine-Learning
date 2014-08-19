"""
Implementation of K-means algorithm
"""

import numpy as np
import pandas as pd
import random
import Auxiliary

def distance2(vector1,vector2):
    """Calculates Cartesian distance between two feature vectors"""
    return sum((vector1[i]-vector2[i])**2 for i in range(len(vector1)))

def find_closest(vector,centers):
    """Finds index of closest center to vector"""
    closestDist = distance2(vector, centers[0])
    closestCenter = 0
    for i in range(len(centers)):
        center = centers[i]
        if distance2(vector,center) < closestDist:
            closestDist = distance2(vector,center)
            closestCenter = i
    return closestCenter

def all_converged(centers,lastCenters):
    """Checks if centers have converged to a stable equilibrium"""
    for i in range(len(centers)):
        if not Auxiliary.has_converged(centers[i],lastCenters[i]):
            return False
    return True

def kmeans(vectors,numCenters):
    """Main method for finding cluster centers"""
    print vectors[0]
    print vectors.shape
    K=numCenters
    numFeatures = len(vectors[0])
    # randomly choose k starting vectors
    centers = []
    indexedVectors = pd.DataFrame(vectors)
    indexedVectors = indexedVectors.apply(lambda x: pd.Series(list(x)))
    closestIndex = [0 for i in range(len(vectors))]
    for i in range(K):
        centers.append(vectors[random.randint(0,len(vectors)-1)])
    print centers
    # repeat algorithm until convergence
    while True:
        lastCenters = centers
        for i in range(len(vectors)):
            closestCenter = find_closest(vectors[i],centers)
            closestIndex[i] = closestCenter
        indexedVectors['Closest Center'] = closestIndex
        groupedVectors = indexedVectors.groupby('Closest Center')
        centers = np.array(groupedVectors.mean())
        print centers
        if all_converged(centers,lastCenters):
            return indexedVectors

irisFile=open("iris.txt","r")
irisArray=[]
for line in irisFile:
    irisArray.append([float(i) for i in line.split(",")[:-2]])
irisArray=np.array(irisArray)
#irisArray=irisArray.reshape([-1,3])
#print kmeans(irisArray,3)

from PIL import Image
im = Image.open("Scorpio.bmp")
p=np.array(im)
p=p.reshape([-1,3])
p=p.astype(float)
comp=kmeans(p,2)
pixels = im.load() # create the pixel map
for i in range(im.size[0]):
    for j in range(im.size[1]):
        #pixels[i,j]=(int(p[j*im.size[0]+i][0]),int(p[j*im.size[0]+i][1]),int(p[j*im.size[0]+i][2]))
        pixels[i,j]=(int(comp.iloc[j*im.size[0]+i][0]),int(comp.iloc[j*im.size[0]+i][1]),int(comp.iloc[j*im.size[0]+i][2]))
im.show()
