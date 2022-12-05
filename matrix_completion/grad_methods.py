from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from utils import make_support


def obj(Mobs, Mask, A, B, mu):
    return 0.5*np.linalg.norm(Mobs - Mask*A.dot(B.T)) + \
           0.5*mu*(np.linalg.norm(A) + np.linalg.norm(B))

def rel_error(Mtrue, Mhat):
    return np.linalg.norm(Mtrue - Mhat)/np.linalg.norm(Mtrue)


def gd(Mtrue, Mobs, Mask, A, B, lamb=10, stepsize=0.5, maxiter=500):

    h = stepsize
    error = [rel_error(M, A.dot(B.T))]
    for k in range(maxiter):
        
        A = A + h*(Mask*(Mobs - A.dot(B.T))).dot(B) - h*lamb*A
        B = B + h*((Mask*(Mobs - A.dot(B.T))).T).dot(A) - h*lamb*B
        
        error.append(rel_error(Mtrue, A.dot(B.T)))

    return error

def nag(Mtrue, Mobs, Mask, A, B, Pa, Pb, lamb=10, stepsize=0.5, mu=0.9, 
        maxiter=500):

    h = stepsize

    error = [rel_error(M, A.dot(B.T))]
    for k in range(maxiter):

        Ahat = A + mu*Pa
        Pa = mu*Pa - h*(-(Mask*(Mobs - Ahat.dot(B.T))).dot(B) + lamb*Ahat)
        A = A + Pa
        
        Bhat = B + mu*Pb
        Pb = mu*Pb - h*(-(Mask*(Mobs - A.dot(Bhat.T))).T.dot(A) + lamb*Bhat)
        B = B + Pb

        error.append(rel_error(Mtrue, A.dot(B.T)))

    return error

def hb(Mtrue, Mobs, Mask, A, B, Pa, Pb, lamb=10, stepsize=0.5, mu=0.9, 
        maxiter=500):

    h = stepsize

    error = [rel_error(M, A.dot(B.T))]
    for k in range(maxiter):

        Pa = mu*Pa - h*(-(Mask*(Mobs - A.dot(B.T))).dot(B) + lamb*A)
        A = A + Pa
        
        Pb = mu*Pb - h*(-(Mask*(Mobs - A.dot(B.T))).T.dot(A) + lamb*B)
        B = B + Pb

        error.append(rel_error(Mtrue, A.dot(B.T)))

    return error

def rgd(Mtrue, Mobs, Mask, A, B, Pa, Pb, lamb=10, stepsize=0.5, mu=0.9, 
        mass=0.5, speed_light=1e4, maxiter=500):

    h = stepsize
    m = mass
    c = speed_light

    error = [rel_error(M, A.dot(B.T))]
    for k in range(maxiter):

        Pa = mu*Pa - h*(-(Mask*(Mobs - A.dot(B.T))).dot(B) + lamb*A)
        A = A + h*c*Pa/np.sqrt(np.linalg.norm(Pa)**2+(m*c)**2)
        
        Pb = mu*Pb - h*(-(Mask*(Mobs - A.dot(B.T))).T.dot(A) + lamb*B)
        B = B + h*c*Pb/np.sqrt(np.linalg.norm(Pb)**2+(m*c)**2)

        error.append(rel_error(Mtrue, A.dot(B.T)))

    return error

def tune_gd(Mtrue, Mobs, Mask, A, B, stepsizerange, lamb=10, maxiter=100):
    bestval = np.inf
    for h in stepsizerange:
        val = gd(Mtrue, Mobs, Mask, A, B, lamb=10, stepsize=h, 
                    maxiter=maxiter)[-1]
        if val < bestval:
            bestval = val
            besth = h
    return bestval, besth

def tune_nag(Mtrue, Mobs, Mask, A, B, Pa, Pb, hrange, murange, lamb,  
              numtrials=100, maxiter=500):
    bestval = np.inf
    for _ in range(numtrials):
        h = np.random.uniform(*hrange)
        mu = np.random.uniform(*murange)
        val = nag(Mtrue, Mobs, Mask, A, B, Pa, Pb, lamb=10, stepsize=h, 
                        mu=mu, maxiter=maxiter)[-1]
        if val < bestval:
            bestval = val
            besth = h
            bestmu = mu
    return bestval, besth, bestmu

def tune_rgd(Mtrue, Mobs, Mask, A, B, Pa, Pb, hrange, murange, mrange,
             crange, lamb, numtrials=100, maxiter=500):
    bestval = np.inf
    for _ in range(numtrials):
        h = np.random.uniform(*hrange)
        mu = np.random.uniform(*murange)
        m = np.random.uniform(*mrange)
        c = np.random.uniform(*crange)
        val = rgd(Mtrue, Mobs, Mask, A, B, Pa, Pb, lamb=10, stepsize=h, 
                        mu=mu, mass=m, speed_light=c, maxiter=maxiter)[-1]
        if val < bestval:
            bestval = val
            besth = h
            bestmu = mu
            bestm = m
            bestc = c
    return bestval, besth, bestmu, bestm, bestc



###############################################################################
if __name__ == '__main__':
    
    from prettytable import PrettyTable
    import sys

    n1,n2 = (500,500) # matrix size
    r = 20 # rank
    sr = 0.2 # sampling ratio

    p = int(sr*n1*n2) # number sampled entries
    d = r*(n1+n2-r) # effective degrees of freedom per measurement
    print(d/p)

    Ml = np.random.normal(0,1,size=(n1,r))
    Mr = np.random.normal(0,1,size=(n2,r))
    M = Ml.dot(Mr.T)
    Om = make_support(M.shape, p) # support of observed entries
    Mobs = Om*M

    rhat = 50
    #A = np.random.rand(n1, rhat)
    A = np.random.normal(0, 1, (n1, rhat))
    #B = np.random.rand(n2, rhat)
    B = np.random.normal(0, 1, (n2, rhat))
    Pa = np.zeros((n1, rhat))
    Pb = np.zeros((n2, rhat))

    mi = 200
    lamb=5
    
    # GD
    h_gd = 8e-3

    # NAG
    h_nag = 6e-3
    mu_nag = 0.8

    # RGD
    h_rgd =  6e-3
    mu_rgd = 0.5
    m_rgd = 6e-3
    c_rgd = 1e3

    fs_gd = gd(M, Mobs, Om, A, B, lamb=lamb, stepsize=h_gd, maxiter=mi)
    fs_nag = nag(M, Mobs, Om, A, B, Pa, Pb, lamb=lamb, stepsize=h_nag, 
                 mu=mu_nag, maxiter=mi)
    fs_rgd = rgd(M, Mobs, Om, A, B, Pa, Pb, lamb=lamb, 
                stepsize=h_rgd,
                mu=mu_rgd,
                mass=m_rgd,
                speed_light=c_rgd,
                maxiter=mi)

    t = PrettyTable()
    t.field_names = ['GD', 'NAG', 'RGD']
    t.add_row(['%.3f'%x 
            for x in [fs_gd[-1], fs_nag[-1], fs_rgd[-1]]])
    print(t)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(fs_gd, label='GD')
    ax.semilogy(fs_nag, label='NAG')
    ax.semilogy(fs_rgd, label='RGD')
    ax.legend(loc=0)
    fig.savefig('objective.pdf')

