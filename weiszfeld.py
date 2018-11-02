# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 12:46:04 2016

Weiszfeld Geometric Median Code

@author: Michael Zhang
"""
#import pdb
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def geometric_median(X, eps = 1e-10, verbose=True, sigma=1e-6):
    """
    X is a M x D matrix (M subsets)
    eps is error tolerance (float)
    """    
    X_dist = np.inf
    assert(len(X.shape) >= 2)
    if len(X.shape) == 3:
        M,N,D = X.shape
    elif len(X.shape) == 2:
        M,D = X.shape
    if M == 1:
        return(X[0])
    else:
        w_star = np.ones(M) * (1./M)
        X_new = X.mean(axis=0)
        
        iters = 0
        while X_dist > eps:
            iters += 1
            X_star = np.copy(X_new)
            w_star = np.power(np.array([ measure_dist(X[m], X_star, sigma) for m in xrange(M)]), -1.)
            w_star /= w_star.sum()
            X_new = np.dot(X.T,w_star).T
            X_dist = measure_dist(X_new, X_star, sigma)
#            if verbose:
#                print("Geom. Med. Iter.: %i\t Current measure distance: %f" % (iters,X_dist))
#                print("Weights: %s" % w_star)
    
        cutoff = 1./(2.*M)
        idx_cutoff = np.where(w_star < cutoff) # truncate any weight below 1/(2M)
        w_star[idx_cutoff] = 0.
        w_star /= w_star.sum()
#        if verbose:
#            print("Geom. Med. Iter.: %i\t Current measure distance: %f" % (iters,X_dist))
#            print("Geometric Median Weights: %s" % w_star)

#        if np.any(w_star < 0):
#            w_star = np.ones(M) * (1./M)
        X_star = np.dot(X.T,w_star).T
#    return((X_star,w_star))
        return(X_star)

                
def measure_dist(X,Y,sigma):
    X_dim = len(X.shape)
    Y_dim = len(Y.shape)
    assert(X_dim == Y_dim)
    assert(X_dim == 1 or X_dim == 2)
    if X_dim == 2:
        N_x,D_x = X.shape
        N_y,D_y = Y.shape
    if X_dim == 1:
        D_x, = X.shape
        D_y, = Y.shape
        N_x = N_y = 1
        X = X.reshape(-1,D_x)
        Y = Y.reshape(-1,D_y)
    assert(D_x == D_y)
#    X_idx = np.triu_indices(N_x,1)    
#    Y_idx = np.triu_indices(N_y,1)    
#    XY_idx = np.triu_indices(N_y,0)    
    X_kernel_dist = (N_x**(-2.)) * (rbf_kernel(X,gamma=sigma).sum())
    Y_kernel_dist = (N_y**(-2.)) * (rbf_kernel(Y,gamma=sigma).sum())    
#    X_kernel_dist = (N_x**(-2.)) * rbf_kernel(X,gamma=sigma)[X_idx].sum()
#    Y_kernel_dist = (N_y**(-2.)) * rbf_kernel(Y,gamma=sigma)[Y_idx].sum()
    XY_kernel_dist = -(2./(N_x*N_y)) * (rbf_kernel(X,Y,gamma=sigma).sum())
    dist_XY = X_kernel_dist + Y_kernel_dist + XY_kernel_dist
    if dist_XY < 0.:
        if np.allclose(dist_XY, 0.):
            print("Distance within tolerance of zero")
            return(np.abs(dist_XY)+ 1e-6)
        else:
            print("Distance is a negative value")
            dist_XY = 1e-6
            return(dist_XY)
#            pdb.set_trace()
#            raise
    else:
        return(dist_XY)
    