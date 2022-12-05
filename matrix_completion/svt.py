"""
Matrix Completion problem using Singular Value Thresholding from
Candes et al 2008.

"""

# author: Guilherme S. Franca <guifranca@gmail.com>
# date: Feb 8, 2019

from __future__ import division

import numpy as np

from utils import soft

def svt(Mobs, Mask, X0, tau=10, stepsize=1, k=None, tol=1e-3, maxiter=500):
    """Singular Value Thresholding algorithm from Cai, Candes, Shen (2008).
    
    We use a sparse approximation to the SVD.
    
    Input
    -----
    
    Mobs : 2D array
        matrix with observed entries
    Mask : 2D array  
        support of observed entries
    tau : float
        constant with the nuclear norm in the objective
    stepsize : float
        stepsize
    k : int
        maximum rank for randomized SVD

    Output
    ------

    Mhat : 2D array
        recoverred matrix
    
    """
    Y = X0
    MobsNorm= np.linalg.norm(Mobs)
    for i in range(maxiter):
        
        X, rank = soft(tau, Y, k=k)
        Y = Y + stepsize*Mask*(Mobs-X)

        error = np.linalg.norm(Mask*(X-Mobs))/MobsNorm
        if error <= tol or i == maxiter:
            break

    return X, rank


###############################################################################
if __name__ == '__main__':
    import data
    from prettytable import PrettyTable
    from utils import make_support
    
    n1,n2 = (40,40)     # matrix size
    r = 2               # rank
    sr = 0.5            # sampling ratio
    p = int(sr*n1*n2)   # number sampled entries
    fr = r*(n1+n2-r)/p  # effective degrees of freedom per measurement
    rm = int(np.floor((n1+n2-np.sqrt((n1+n2)**2 - 4*p))/2)) # maximum rank
    
    Ml = np.random.normal(size=(n1,r))
    Mr = np.random.normal(size=(n2,r))
    M = Ml.dot(Mr.T)
    Om = make_support(M.shape, p) # support of observed entries
    Mobs = Om*M
    X0 = np.zeros(shape=M.shape)

    tau = 5*(n1+n2)/2
    delta = 1.2*n1*n2/p
    
    X, rank = svt(Mobs,Om,X0,tau=tau,stepsize=delta,k=rm,tol=1e-8,maxiter=500)
    rel_error = np.linalg.norm(X-M)/np.linalg.norm(M)

    t = PrettyTable()
    t.field_names = ['(n1,n2)', 'rank', 'sampling ratio', 'num. meas.', 
                     'deg./meas.', 'max rank']
    t.add_row([(n1,n2), r, sr, p, fr, rm])
    print(t)
    
    t = PrettyTable()
    t.field_names = ['rel. error', 'estimated rank']
    t.add_row(['%.2E'%rel_error, rank])
    print(t)
    
