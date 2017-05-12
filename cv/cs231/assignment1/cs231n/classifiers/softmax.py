import numpy as np
import math
import pdb
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    denom = np.sum(math.e**np.dot(X[i],W))
    loss += -np.dot(X[i],W[:,y[i]])+np.log(denom)
    for j in range(num_classes):
      dW[:,j] += X[i]*math.e**np.dot(X[i],W[:,j])/denom
      if j == y[i]:
        dW[:,j] -= X[i]
  loss /= num_train
  loss += reg*np.sum(W**2)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = np.dot(X,W)
  denom = np.sum(math.e**scores,axis=1)
  loss = -np.sum(scores[range(num_train),y])+np.sum(np.log(denom))
  loss /= num_train
  loss += reg*np.sum(W**2)
  mask = np.zeros((num_train,num_classes))
  mask[range(num_train),y] = 1
  dW = -np.dot(X.T,mask)+np.dot(X.T,math.e**scores/denom.reshape((num_train,1)))
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

