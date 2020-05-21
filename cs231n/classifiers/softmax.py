from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    n_batches = len(y)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss = 0
    
    for i, (Xi, yi) in enumerate(zip(X, y)):
        p = W.T.dot(Xi) # unnormalized log probabilities
        p -= np.max(p) # shift values to handle numeric instability
        f = np.exp(p) / np.sum(np.exp(p)) # softmax function
        loss += -np.log(f[yi])
        
        for j, fj in enumerate(f):
            if j == yi:
                dW[:, j] += Xi * (fj - 1)
            else:
                dW[:, j] += Xi * fj

    loss /= n_batches
    loss += reg * np.sum(W ** 2)
    
    dW /= n_batches
    dW += reg * 2 * W 
        
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    n_batches = len(y)
    cls_idx = y, np.arange(n_batches)
    
    p = W.T.dot(X.T)    
    p -= np.max(p, axis=0)
    e = np.exp(p)
    f = e / np.sum(e, axis=0) 
    loss = np.average(-np.log(f[cls_idx])) + reg * np.sum(W ** 2)
    
    f[cls_idx] -= 1
    dW[:] = X.T.dot(f.T) / n_batches + reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
