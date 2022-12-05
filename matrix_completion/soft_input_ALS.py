from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from utils import make_support


def obj(Mobs, Mask, A, B, mu):
    return 0.5*np.linalg.norm(Mobs - Mask*A.dot(B.T)) + \
           0.5*mu*(np.linalg.norm(A) + np.linalg.norm(B))

def rel_error(Mtrue, Mhat):
    return np.linalg.norm(Mtrue - Mhat)/np.linalg.norm(Mtrue)

def softImputeALS(Mtrue, Mobs, Mask, U, V, lamb=1, maxiter=100):
    r = U.shape[1]
    D = np.eye(r)
    A = U.dot(D)
    B = V.dot(D)

    error = [rel_error(Mtrue, A.dot(B.T))]
    for k in range(maxiter):
       
        # compute B
        Xstar = Mobs - Mask*(A.dot(B.T)) + A.dot(B.T)
        BtildeT = np.linalg.inv(D**2+lamb*np.eye(r)).dot(D.dot(U.T.dot(Xstar)))
        Utilde, dsquare, VtildeT = np.linalg.svd(BtildeT.T.dot(D))
        V = Utilde
        D = np.diag(np.sqrt(dsquare))
        B = V.dot(D)

        # compute A
        XstarT = (Mobs - Mask*(A.dot(B.T)) + A.dot(B.T))
        Xstar = XstarT.T
        AtildeT = np.linalg.inv(D**2+lamb*np.eye(r)).dot(D.dot(V.T.dot(Xstar)))
        Utilde, dsquare, VtildeT = np.linalg.svd(AtildeT.T.dot(D))
        D = np.diag(np.sqrt(dsquare))
        V = Utilde
        A = V.dot(D)

        Mhat = A.dot(B.T)
        error.append(rel_error(Mtrue, Mhat))

    # compute the rank
    #V = V.dot(Rt.T)
    #Dsigma_threshold = np.diag(np.max(dsigma - lamb, 0))
    #return U, V, Dsiamg_threshold
    
    return error



if __name__ == '__main__':
    
    from prettytable import PrettyTable

    n1,n2 = (100,100) # matrix size
    r = 5 # rank
    sr = 0.4 # sampling ratio

    p = int(sr*n1*n2) # number sampled entries
    d = r*(n1+n2-r) # effective degrees of freedom per measurement
    print(d/p)

    Ml = np.random.normal(0,1,size=(n1,r))
    Mr = np.random.normal(0,1,size=(n2,r))
    M = Ml.dot(Mr.T)
    Om = make_support(M.shape, p) # support of observed entries
    Mobs = Om*M

    rhat = 300
    #A = np.random.rand(n1, rhat)
    A = np.random.normal(0, 0.1, (n1, rhat))
    #B = np.random.rand(n2, rhat)
    B = np.random.normal(0, 0.1, (n2, rhat))
    Pa = np.zeros((n1, rhat))
    Pb = np.zeros((n2, rhat))

    mi = 1000
    lamb=0.5
    
    fs_nag = nag(M, Mobs, Om, A, B, Pa, Pb, lamb=lamb, stepsize=h_nag, 
                 mu=mu_nag, maxiter=mi)
    fs_hb = hb(M, Mobs, Om, A, B, Pa, Pb, lamb=lamb, stepsize=h_hb, 
                 mu=mu_hb, maxiter=mi)
    fs_rgd = rgd(M, Mobs, Om, A, B, Pa, Pb, lamb=lamb, stepsize=h_rgd, 
                 mu=mu_rgd, mass=m, speed_light=c, maxiter=mi)

    t = PrettyTable()
    t.field_names = ['GD', 'NAG', 'HB', 'RGD']
    t.add_row(['%.3f'%x 
            for x in [fs_gd[-1], fs_nag[-1], fs_hb[-1], fs_rgd[-1]]])
    print(t)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(fs_gd, label='GD')
    ax.semilogy(fs_nag, label='NAG')
    ax.semilogy(fs_hb, label='HB')
    ax.semilogy(fs_rgd, label='RGD')
    ax.legend(loc=0)
    fig.savefig('objective.pdf')



