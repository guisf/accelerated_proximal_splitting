"""
Non-negative matrix completion using MovieLens dataset.
"""

from __future__ import division

import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage import color
from skimage import io

from three_split import davis_yin, e2admm, eadmm
from utils import make_support


tau = 30
stepsize = 0.8
lower = 1
upper = 5
rank = 50
pickle_fname = 'data/movielens1M.pickle'

algorithms = [davis_yin, e2admm, eadmm]
acceleration = [None, 'nest', 'hb']
damping = [0, 3, 0.3]
names = ['DY', 'E2-ADMM', 'E-ADMM']
variations = ['', ' (Nest)', ' (HB)']
plot_algs = ['E-ADMM', 'E-ADMM (Nest)', 'E-ADMM (HB)']
markers = ['v', '^', 'o']

args = [tau, stepsize, lower, upper]
#kargs = {'rank': rank, 'obs_only': True,  'maxiter': 500, 'tol' :1e-3,
#         'conv_error': True, 'iteronly': False, 'randsvd': False}
kargs = {'obs_only': True,  'maxiter': 500, 'tol' :1e-3,
         'conv_error': True, 'iteronly': False, 'randsvd': False}


def algo(M, omega, func, accel, damping):
    """Function to call algorithm with parameters. Return a string
    with the results.
    
    """
    (Mhat, rel_error), time = func(M, omega, *args, **kargs,
                                   accel=accel, damping=damping)
    return Mhat, rel_error, time

def convergence(M, omega):
    """Create a pickle file with the true reconstruction error
    for each algorithm.

    """
    results = {}
    for var, damp, accel in zip(variations, damping, acceleration):
        for name, algorithm in zip(names, algorithms):
            full_name = name+var
            _, rel_error, time = algo(M, omega, algorithm, accel, damp)
            results[full_name] = [rel_error, time]
            print(full_name, rel_error[-1], time)
    pickle.dump(results, open(pickle_fname, 'wb'))
    return results

def plot_convergence(ax):
    """Make the convergence plot of the above function."""
    results = pickle.load(open(pickle_fname, 'rb'))
    i = 0
    for algname, val in results.items():
        rate, time = val
        rate = np.array(rate)
        #ax.plot(np.linspace(0, time, len(rate)), rate,
        #        label=algname, marker=markers[i%3], markevery=int(time/15))
        num = len(rate)
        ax.plot(range(num), rate,
                label=algname, marker=markers[i%3], markevery=int(0.15*num))
        i += 1

def make_plot_convergence():
    """Create the convergence plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    plot_convergence(ax)
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.legend(loc=0)
    #ax.set_xlabel('time (s)')
    ax.set_xlabel('iteration')
    ax.set_ylabel(r'relative error')
    ax.set_xlim([0,30])
    ax.set_ylim([0.2,1])
    fig.savefig('movielens10m_rate.pdf', bbox_inches='tight')
    
def load_movielens10m():
    # putting rates in a 2d matrix
    data = np.array(np.genfromtxt('data/ml-1m/ratings.dat', delimiter="::"))
    ratings = np.zeros((6040,3952))
    for line in data:
        ratings[int(line[0])-1,int(line[1])-1]=line[2]

    #create support matrix
    omega = np.zeros(ratings.shape, dtype=int)
    omega[np.where(ratings>0)] = 1

    return ratings, omega

def load_movielens10m_v2():
    # putting rates in a 2d matrix
    data = np.array(np.genfromtxt('data/ml-1m/ratings.dat', delimiter="::"))
    ratings = np.zeros((6040,3952))
    for line in data:
        ratings[int(line[0])-1,int(line[1])-1]=line[2]

    #create support matrix
    true_omega = np.zeros(ratings.shape, dtype=int)
    true_omega[np.where(ratings>0)] = 1
    
    # we are gonna subsample this matrix 
    sample_rate = 0.1
    number_elements = int(sample_rate*ratings.shape[0]*ratings.shape[1])
    sub_omega = make_support(ratings.shape, number_elements)

    omega = true_omega - sub_omega

    return ratings, omega, sub_omega



###############################################################################
if __name__ == '__main__':

    #ratings, omega = load_movielens10m()
    #convergence(ratings, omega)
    #make_plot_convergence()
    
    ratings, omega, sub_omega = load_movielens10m_v2()
    sub_ratings = omega*ratings
    ratings_hat, rel_error, time = algo(sub_ratings, omega, eadmm, 'hb', 0.3)
    eval_error = np.linalg.norm(sub_omega*(
                    ratings_hat-ratings))/np.linalg.norm(sub_omega*ratings)
    print(rel_error[-1], eval_error, time)

