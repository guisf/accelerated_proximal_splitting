"""
Nonnegative Matrix Completion using forward-backward and Tseng splitting.
    
"""

# author: Guilherme S. Franca <guifranca@gmail.com>
# date: July, 2019

from __future__ import division

import numpy as np

from utils import soft, warm, accelerate


def fb(Y, Mask, X0, mu=10, stepsize=1, k=None, maxiter=500, tol=1e-6,
       accel=None, damp=3):
    """Forward-backward splitting method."""
    h = stepsize
    X = X0
    Xhat = X0
    i = 0
    converged = False
    while not converged and i < maxiter:
        oldX = X

        X, rank = soft(mu*h, Xhat-h*(Mask*Xhat-Y), k=k)
        Xhat = accelerate(X, oldX, i, h, method=accel, damp=damp)

        error = np.linalg.norm(X-oldX)/np.max([1, np.linalg.norm(oldX)])
        if error <= tol:
            converged = True
        i += 1
    return X, rank

def tseng(Mobs, Mask, X0, mu=10, stepsize=1, k=None, maxiter=500, tol=1e-6,
          accel=None, damp=3):
    """Tseng or forward-backward-forward method."""
    h = stepsize
    X = X0
    Xhat = X0
    i = 0
    converged = False
    while not converged and i < maxiter:
        oldX = X

        X, rank = soft(mu*h, Xhat-h*(Mask*Xhat-Mobs), k=k)
        X = X - h*Mask*(X - Xhat)
        Xhat = accelerate(X, oldX, i, h, method=accel, damp=damp)

        error = np.linalg.norm(X-oldX)/np.max([1, np.linalg.norm(oldX)])
        if error <= tol:
            converged = True
        i += 1
    return X, rank


##############################################################################
if __name__ == '__main__':
    
    from prettytable import PrettyTable
    import time
    import sys
    
    import data
    from utils import make_support
    
    n1,n2 = (1000,1000)     # matrix size
    r = 10                  # rank
    sr = 0.12               # sampling ratio

    p = int(sr*n1*n2)   # number sampled entries
    fr = r*(n1+n2-r)/p  # effective degrees of freedom per measurement
    rm = int(np.floor((n1+n2-np.sqrt((n1+n2)**2 - 4*p))/2)) # maximum rank
    
    Ml = np.random.normal(size=(n1,r))
    Mr = np.random.normal(size=(n2,r))
    M = Ml.dot(Mr.T)
    Om = make_support(M.shape, p) # support of observed entries
    Mobs = Om*M
    X0 = np.zeros(shape=M.shape)
    
    print("* Problem Data")
    t = PrettyTable()
    t.field_names = ['(n1,n2)', 'true rank', 'sampling ratio', 'num meas', 
                     'deg/meas', 'maximum rank']
    t.add_row([(n1,n2), r, sr, p, fr, rm])
    print(t)

    # single run #############################################################
    # acceleration improves considerably for a single run
    a = time.time()
    X, rank = fb(Mobs, Om, X0, mu=3, stepsize=1, k=20, maxiter=500, tol=1e-4,
                accel='hb', damp=-np.log(0.9))
    b = time.time()-a
    error = np.linalg.norm(X-M)/np.linalg.norm(M)
    print(error, rank, b)
    sys.exit()

    ##########################################################################
    # multiple runs
    # acceleration does not improve under "annealing" and warm start
    #
    #mubar = 1e-8
    mubar = 1e-3
    #mu = (1./4.)*np.linalg.norm(Mobs)
    mu = 30
    mus = []
    while mu > mubar:
        mu = np.max([mu/4., mubar])
        mus.append(mu)
    t = PrettyTable()
    t.field_names = ['algorihtm', 'relative error', 'time (s)', 'rank']
    algorithms = [fb]#, tseng]
    acceltype = [None, 'nest', 'hb']
    dampings = [1, 3, -np.log(0.9)]
    #kargs = {'stepsize':1, 'k':rm, 'maxiter':500, 'tol':1e-10}
    kargs = {'stepsize':1, 'k':20, 'maxiter':500, 'tol':1e-6}
    for algorithm in algorithms:
        for accel, damp in zip(acceltype, dampings):
            a = time.time()
            X, ranks = warm(algorithm, Mobs, Om, mus, **kargs)
            b = time.time()-a
            error = np.linalg.norm(X-M)/np.linalg.norm(M)
            t.add_row(['%s (%s)'%(algorithm.__name__,accel),'%.2E'%error,
                        b,ranks[-1]])
    print('* Results')
    print(t)
    

