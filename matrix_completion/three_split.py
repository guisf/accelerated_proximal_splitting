"""
Nonnegative Matrix Completion using 3 operator splitting.
    
Problem:

$$
\min_X \tau \| X \|_* + (1/2)\| P_\Omega(X-M)\|_F^2 \quad
\mbox{subject to  $l \le X \le u$}.
$$

We use  the general formulation

$$
\min f(x) + g(x) + w(x)
$$

where the function g(x) = I_c(x) is the projection into a box
of sizes [u, l].


Basic notation:

$\Omega$ represents a support matrix having 1 at the observed entries and 0
elsewhere. $M$ is the matrix we attempt to recover.


"""

# author: Guilherme S. Franca <guifranca@gmail.com>
# date: Feb 8, 2019

from __future__ import division

import numpy as np

from utils import soft, warm, accelerate, clip


def davis_yin(Mobs, Mask, X0, mu=10, stepsize=0.5, l1=0, l2=100, k=None, 
              accel=None, damp=3, tol=1e-4, maxiter=500):
    """
    Davis-Yin three operator splitting, including new accelerated variants.

    References:
    "A Three-Operator Splitting Scheme and its Optimization Applications",
    Davis and Yin
    
    """
    h = stepsize
    Xhat = X0
    X = X0
    i = 0
    converged = False
    Xs = []
    ranks = []
    while not converged and i < maxiter:
        oldX = X
        
        X14, rank = soft(mu*h, Xhat, k=k)
        X12 = 2*X14 - Xhat
        X34 = clip(X12 - h*(Mask*X14-Mobs), l1, l2)
        X = Xhat + X34 - X14
        Xhat = accelerate(X, oldX, i, h, method=accel, damp=damp)
    
        Xs.append(X)
        ranks.append(rank)
        
        error = np.linalg.norm(X-oldX)/np.max([1, np.linalg.norm(oldX)])
        if error <= tol:
            converged = True
        i += 1
    return Xs, ranks
    #return X, rank

def davis_yin2(Mobs, Mask, X0, mu=10, stepsize=0.5, l1=0, l2=100, k=None, 
              accel=None, damp=3, tol=1e-4, maxiter=500):
    """
    Davis-Yin three operator splitting, including new accelerated variants.
    We change the order of the proximal operators.

    """
    h = stepsize
    Xhat = X0
    X = X0
    i = 0
    converged = False
    while not converged and i < maxiter:
        oldX = X

        X14 = clip(Xhat, l1, l2)
        X12 = 2*X14 - Xhat
        X34, rank = soft(mu*h, X12 - h*(Mask*X14-Mobs), k=k)
        X = Xhat + X34 - X14
        Xhat = accelerate(X, oldX, i, h, method=accel, damp=damp)
        
        error = np.linalg.norm(X-oldX)/np.max([1, np.linalg.norm(oldX)])
        if error <= tol:
            converged = True
        i += 1
    return X, rank

def eadmm(Mobs, Mask, X0, mu=10, stepsize=0.5, l1=0, l2=100, k=None, 
          accel=None, damp=3, tol=1e-4, maxiter=500):
    """Extended ADMM algorithm for matrix completion problem."""
    h = stepsize
    Xhat = X0
    X = X0
    C = np.zeros(X.shape)
    i = 0
    converged = False
    Xs = []
    ranks = []
    while not converged and i < maxiter:
        oldX = X

        X12, rank = soft(mu*h, Xhat-h*(Mask*Xhat-Mobs)+h*C, k=k)
        X = clip(X12-h*C, l1, l2)
        C = C+(1/h)*(X-X12)
        Xhat = accelerate(X, oldX, i, h, method=accel, damp=damp)
        
        Xs.append(X)
        ranks.append(rank)
        
        error = np.linalg.norm(X-oldX)/np.max([1, np.linalg.norm(oldX)])
        if error <= tol:
            converged = True
        i += 1
    return Xs, ranks
    #return X12, rank

def fastadmm(Mobs, Mask, X0, mu=10, stepsize=0.5, l1=0, l2=100, k=None,
          accel=None, damp=3, tol=1e-4, maxiter=500):
    """Fast ADMM where we also accelerate the dual variable."""
    h = stepsize
    Xhat = X0
    X = X0
    C = np.zeros(X.shape)
    Chat = C
    i = 0
    converged = False
    Xs = [] 
    ranks = []
    while not converged and i < maxiter:
        oldX = X
        oldC = C

        X12, rank = soft(mu*h, Xhat-h*(Mask*Xhat-Mobs)+h*Chat, k=k)
        X = clip(X12-h*Chat, l1, l2)
        C = Chat+(1/h)*(X-X12)
        Xhat = accelerate(X, oldX, i, h, method=accel, damp=damp)
        Chat = accelerate(C, oldC, i, h, method=accel, damp=damp)

        Xs.append(X)
        ranks.append(rank)

        error = np.linalg.norm(X-oldX)/np.max([1, np.linalg.norm(oldX)])
        if error <= tol:
            converged = True
        i += 1
    return Xs, ranks
    #return X12, rank


def e2admm(Mobs, Mask, X0, mu=10, stepsize=0.5, l1=0, l2=100, k=None, 
           accel=None, damp=3, tol=1e-4, maxiter=500):
    """Extended ADMM algorithm version 2 for matrix completion problem."""
    h = stepsize
    Xhat = X0
    X = X0
    C = np.zeros(X.shape)
    i = 0
    converged = False
    Xs = []
    ranks = []
    while not converged and i < maxiter:
        oldX = X

        X12, rank = soft(mu*h, Xhat-h*(Mask*Xhat-Mobs)+h*C, k=k)
        X = clip(X12-h*C, l1, l2)
        C = C+(1/h)*(0.5*(X+oldX)-X12)
        Xhat = accelerate(X, oldX, i, h, method=accel, damp=damp)
        
        Xs.append(X)
        ranks.append(rank)
        
        error = np.linalg.norm(X12-oldX)/np.max([1, np.linalg.norm(oldX)])
        if error <= tol:
            converged = True
        i += 1
    return Xs, ranks
    #return X, rank


##############################################################################
if __name__ == '__main__':

    from prettytable import PrettyTable
    import time
    import sys
    
    import data
    from utils import make_support

    n1,n2 = (100,100)     # matrix size
    r = 5                  # rank
    sr = 0.4                # sampling ratio

    p = int(sr*n1*n2)   # number sampled entries
    fr = r*(n1+n2-r)/p  # effective degrees of freedom per measurement
    rm = int(np.floor((n1+n2-np.sqrt((n1+n2)**2 - 4*p))/2)) # maximum rank

    Ml = np.random.normal(0,1,size=(n1,r))
    Mr = np.random.normal(0,1,size=(n2,r))
    M = Ml.dot(Mr.T)
    #M = np.abs(M)
    Om = make_support(M.shape, p) # support of observed entries
    Mobs = Om*M
    X0 = np.zeros(shape=M.shape)
    print(M.min(), M.max())

    print("* Problem Data")
    t = PrettyTable()
    t.field_names = ['(n1,n2)', 'true rank', 'samp ratio', 'num meas',
                     'deg/meas', 'max rank']
    t.add_row([(n1,n2), r, sr, p, '%.3f'%fr, rm])
    print(t)

    # single run #############################################################
    # acceleration improves considerably for a single run
    t = PrettyTable()
    t.field_names = ['algorithm', 'rel error', 'rank', 'time']
    algorithms = [davis_yin, eadmm, e2admm]
    accelerations = [None, 'nest', 'hb']
    dampings = [1, 3, -np.log(0.9)]
    for algo in algorithms:
        for accel, damp in zip(accelerations, dampings):
            a = time.time()
            X, rank = algo(Mobs, Om, X0, mu=0.5, stepsize=1, k=None, l1=-30, 
                    l2=30, maxiter=1000, tol=1e-4, accel=accel, damp=damp)
            b = time.time()-a
            error = np.linalg.norm(X[-1]-M)/np.linalg.norm(M)
            t.add_row(['%s (%s)'%(algo.__name__,accel), 
                       '%.3f'%error, rank[-1], '%.3f'%b])
    print(t)
    
