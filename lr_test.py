import numpy as np
from numpy import *
from plotBoundary import *
from matplotlib import pylab
import pylab as pl
from sklearn import *
#from sklearn import linear_model

# import your LR training code

# parameters
name = '2'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

def train_logistic(penalty, C, tol):
# Carry out training.
	clf = linear_model.LogisticRegression(penalty=penalty, C=C, tol=tol)
	clf.fit(X,Y.reshape((Y.shape[0])))

	print 'Intercept is {1}, coefficients are {0}'.format(clf.coef_, clf.intercept_[0])
	print 'C is {0}, tolerance is {1}, number of iterations is {2}'.format(C, tol, clf.n_iter_[0])
	return clf


# Define the predictLR(x) function, which uses trained parameters
def predictLR(x):
	ret = clf.predict_proba(x)[0][1]
	return ret

'''
for C in [1e10, 1]:
	for tol in [1e-2,1e-3,1e-4,1e-5,1e-6]:
		clf = train_logistic('l2', C, tol)
'''

'''
for penalty in ['l2', 'l1']:
	for C in [1e10, 1e7, 1e4, 1]:
		clf = train_logistic(penalty, C, 1e-4)
		# plot training results
		plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train ({0} penalty, C={1})'.format(penalty, C))
		pl.show()
		clf.score(X, Y)
'''

clf = train_logistic('l2', 1, 1e-4)
# plot training results
#plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train ({0} penalty, C={1})'.format('l2', 1e10))
#pl.show()
print 'Training error is {0}'.format(clf.score(X, Y))

#'''
print '======Validation======'
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

print 'Validation error is {0}'.format(clf.score(X, Y))

# plot validation results
#plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
#pl.show()
#'''

test = loadtxt('data/data'+name+'_test.csv')
X = test[:,0:2]
Y = test[:,2:3]

print 'Test error is {0}'.format(clf.score(X, Y))
