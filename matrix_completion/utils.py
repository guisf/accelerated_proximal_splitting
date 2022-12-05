# author: Guilherme S. Franca <guifranca@gmail.com>
# date: July, 2019

from __future__ import division

import numpy as np
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd


def soft(mu, X, k=None):
    """Soft threshold SVD operator.
    
    Parameters
    ----------
    mu : float
        constant to threshold
    X : 2D array
        Input matrix to compute SVD
    k : int
        maximum rank for SVD computation
    
    Output
    ------
    Xs : 2D array
        array containing the softhreshold operation
    rank : int
        we also return the estimated rank
    
    """
    if k == None:
        U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    else:
        U, sigma, Vt = svds(X, k=k)
    sigma = np.maximum(sigma-mu, 0)
    rank = len(sigma[sigma>0])
    return U.dot(np.diag(sigma).dot(Vt)), rank

def warm(algorithm, Y, Mask, mus, **kargs):
    """Several warm starts with a list of mu's (decreasing).
    
    Parameters
    ----------
    algorithm : function
        the algorithm to be used. It must accept arguments
        in the order algorithm(Y, Mask, mu, **kargs)
        The output of the algorithm should be (X, rank)
    Y : 2D array
        matrix with observed entries
    Mask : 2D array
        support of observed entries
    mus : list
        list of mus
    kargs : all the other named arguments to use in algorithm

    Output
    ------
    (X, ranks) where X is the final completed matrix and
    ranks is a list with the rank estimate for each mu
    
    """
    X = np.zeros(Y.shape)
    X_history = []
    rank_history = []
    for mu in mus:
        Xs, ranks = algorithm(Y, Mask, X, mu=mu, **kargs)
        X = Xs[-1]
        rank = ranks[-1]
        X_history.append(Xs)
        rank_history.append(ranks)
    return X, rank, X_history, rank_history

def accelerate(X, oldX, i, h, method=None, damp=3):
    """Return accelerated variable (or not).
    Parameters
    ----------
    X : 2D array
        current solution estimate
    oldX : 2D array
        previous solution estimate
    i : int
        current iteration
    h : float
        stepsize
    method : {None, 'nest', 'hb}
        type of acceleration
    damp : float
        damping factor value

    Output
    ------
    Xhat : 2D array
        new accelerated variable
    
    """
    if method == 'nest':
        w = i/(i+damp)
    elif method == 'hb':
        w = 1-damp*np.sqrt(h)
    else:
        w = 0
    return X+w*(X-oldX)
    
def clip(x, l1, l2):
    """Projections into a box."""
    return np.maximum(l1, np.minimum(x, l2))

def objective(M, Mhat, omega, tau):
    """Objective function."""
    f1 = tau*np.linalg.norm(Mhat, ord='nuc')
    f2 = 0.5*(np.linalg.norm(omega*(Mhat-M))**2)
    return f1+f2

def make_support(shape, m):
    """Creates support for a matrix with uniform distribution.
   
    Input
    -----
    shape: (rows, columns)
        m: number of nonzero entries

    Output
    ------
    Boolean array of dimension ``shape'' with True values on nonzero positions
    
    """
    total = shape[0]*shape[1]
    omega = np.zeros(total, dtype=int)
    ids = np.random.choice(range(total), m, replace=False)
    omega[ids] = 1
    omega = omega.reshape(shape)
    return omega


##############################################################################
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from three_split import davis_yin
    
    # testing the annealing procedure

    r = 3
    A = np.random.normal(size=(100,r))
    B = np.random.normal(size=(100,r))
    M = A.dot(B.T)
    X0 = np.zeros(M.shape)

    sr = 0.3
    p = int(sr*np.prod(M.shape))
    fr = r*(np.sum(M.shape)-r)/p
    Mask = make_support(M.shape, p)
    Mobs = Mask*M

    print(fr)

    ## choose annealing schedule #######
    eta = 0.25
    mu = eta*np.linalg.norm(Mobs)
    mubar = 1e-8
    mus = []
    while mu > mubar:
        mu = np.max([eta*mu, mubar])
        mus.append(mu)
    ###################################
    
    X, rank, Xhist, rankhist = warm(davis_yin,Mobs,Mask,mus=mus,
        stepsize=1,k=30,l1=-15,l2=15,tol=1e-6,maxiter=500)
    errors1 = [np.linalg.norm(x-M)/np.linalg.norm(M) for hist in Xhist for x in hist]
    print(np.linalg.norm(X-M)/np.linalg.norm(M), rank)
    
    X, rank, Xhist, rankhist = warm(davis_yin,Mobs,Mask,mus=mus,
        stepsize=1,k=30,l1=-15,l2=15,tol=1e-6,maxiter=500,accel='nest',damp=3)
    errors2 = [np.linalg.norm(x-M)/np.linalg.norm(M) for hist in Xhist for x in hist]
    print(np.linalg.norm(X-M)/np.linalg.norm(M), rank)
    
    X, rank, Xhist, rankhist = warm(davis_yin,Mobs,Mask,mus=mus,
        stepsize=1,k=30,l1=-15,l2=15,tol=1e-6,maxiter=500,accel='hb',damp=-np.log(0.9))
    errors3 = [np.linalg.norm(x-M)/np.linalg.norm(M) for hist in Xhist for x in hist]
    print(np.linalg.norm(X-M)/np.linalg.norm(M), rank)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.plot(errors1)
    ax.plot(errors2)
    ax.plot(errors3)
    plt.show()

