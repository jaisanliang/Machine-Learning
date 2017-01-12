"""
Module for doing linear regression using both
batch and stochastic gradient descent to find feature weights
"""

import Auxiliary

d = 3
eta = 0.001

class LinearRegression(Classifier):
    def __init__(self):
        self.w = None

    def vector_cost(self,x,y):
        """Calculates difference between actual and predicted (given feature vector) value"""
        y_p = np.dot(x,self.w)
        return y-y_p

    def batchDescentStep(self,X,Y):
        """Calculates new feature weights after one batch descent step"""
        w_new = np.copy(self.w)
        for i in range(self.d):
            grad = sum(self.vector_cost(X[j],Y[j])*X[j][i] for j in range(len(vectors)))
            w_new[i] += eta*grad
        w = w

    def stochasticDescentStep(self,x,y):
        """Calculates new feature weights after one stochastic descent step"""
        w_new = np.copy(w)
        for i in range(d):
            grad = (y-np.dot(x,self.w))*x[i]
            w_new[i] += eta*grad
        return newWeights

    def train(self,X,Y,batch=True):
        """
        Main method for calculating weights of features
        Vector is feature vector with correct classification appended
        """
        self.d = len(X[0])
        n = len(X)
        self.w = np.zeros(d)
        step = 0
        while True:
            w_prev = np.copy(self.w)
            if batch:
                self.batchDescentStep(X,Y)
            else:
                self.stochasticDescentStep(X[step],Y[step])
                step = (step+1) % n
            if Auxiliary.converged(self.w,w_prev):
                return

class LOESS(Classifier):
    '''
    Locally weighted linear regression
    TODO: implement
    Implement Fisher scoring/Newton's method?
    '''
    def __init__(self):
        self.w = None

    def vector_cost(self,x,y):
        """Calculates difference between actual and predicted (given feature vector) value"""
        y_p = np.dot(x,self.w)
        return y-y_p

    def batchDescentStep(self,X,Y):
        """Calculates new feature weights after one batch descent step"""
        w_new = np.copy(self.w)
        for i in range(self.d):
            grad = sum(self.vector_cost(X[j],Y[j])*X[j][i] for j in range(len(vectors)))
            w_new[i] += eta*grad
        w = w

    def stochasticDescentStep(self,x,y):
        """Calculates new feature weights after one stochastic descent step"""
        w_new = np.copy(w)
        for i in range(d):
            grad = self.vector_cost(x,y)*x[i]
            w_new[i] += eta*grad
        return newWeights

    def train(self,X,Y,batch=True):
        """
        Main method for calculating weights of features
        Vector is feature vector with correct classification appended
        """
        self.d = len(X[0])
        n = len(X)
        self.w = np.zeros(d)
        step = 0
        while True:
            w_prev = np.copy(self.w)
            if batch:
                self.batchDescentStep(X,Y)
            else:
                self.stochasticDescentStep(X[step],Y[step])
                step = (step+1) % n
            if Auxiliary.converged(self.w,w_prev):
                return
