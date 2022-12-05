"""
Image recovery using 3 operator splitting.
"""

import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage import color
from skimage import io

from three_split import davis_yin, eadmm
from utils import make_support

#imgfile = 'data/jaqueline.jpg'
#imgfile = 'data/dali2.jpg'
imgfile = 'data/van_gogh6.jpg'
tau = 0.5
stepsize = 0.8
lower = 0
upper = 1
rank = 100
pickle_fname = 'data/image_recovery.pickle'

algorithms = [davis_yin, eadmm]
acceleration = [None, 'nest', 'hb']
damping = [0, 3, 0.3]
names = ['Davis-Yin', 'ADMM']
variations = ['', ' decaying', ' constant']
plot_algs = ['E-ADMM', 'E-ADMM (Nest)', 'E-ADMM (HB)']
markers = ['v', '^', 'o']

img = color.rgb2gray(io.imread(imgfile))

args = [tau, stepsize, lower, upper]
#kargs = {'rank': rank, 'obs_only': False,  'maxiter': 500, 'tol' :1e-4,
#         'conv_error': True, 'iteronly': False, 'randsvd': False}
kargs = {'maxiter': 500, 'tol' :1e-4}

def algo(img, omega, func, accel, damping):
    """Function to call algorithm with parameters. Return a string
    with the results.
    
    """
    (Mhats, ranks) = func(img, omega, *args, **kargs,
                          accel=accel, damping=damping)
    return Mhats, ranks

def convergence(sample_rate):
    """Create a pickle file with the true reconstruction error
    for each algorithm.
    
    """
    number_elements = int(sample_rate*img.shape[0]*img.shape[1])
    omega = make_support(img.shape, number_elements)

    results = {}
    for var, damp, accel in zip(variations, damping, acceleration):
        for name, algorithm in zip(names, algorithms):
            full_name = name+var
            _, rel_error, time = algo(img, omega, algorithm, accel, damp)
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
        ax.plot(np.linspace(0, time, len(rate[1:])), rate[1:], 
                label=algname, marker=markers[i%3], markevery=int(0.2*time))
        i += 1

def make_plot_convergence():
    """Create the convergence plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    plot_convergence(ax)
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.legend(loc=0)
    ax.set_xlabel('time (s)')
    ax.set_ylabel(r'recovery error')
    ax.set_xlim([0,90])
    #ax.set_ylim([0.19,1])
    fig.savefig('image_recovery_rate.pdf', bbox_inches='tight')

def plot_rank():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    plot_convergence(ax)
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.legend(loc=0)
    ax.set_xlabel('time (s)')
    ax.set_ylabel(r'recovery error')
    ax.set_xlim([0,90])
    #ax.set_ylim([0.19,1])
    fig.savefig('image_recovery_rate.pdf', bbox_inches='tight')
    

def reconstruct_image():
    sample_rate = 0.3
    number_elements = int(sample_rate*img.shape[0]*img.shape[1])
    omega = make_support(img.shape, number_elements)

    img_hat, rel_error, time = algo(img, omega, eadmm, None, 0.3)
    print(rel_error[-1], time)
    img_hat, rel_error, time = algo(img, omega, eadmm, 'hb', 0.3)
    print(rel_error[-1], time)
    img_hat, rel_error, time = algo(img, omega, eadmm, 'nest', 3)
    print(rel_error[-1], time)

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, 
                        figsize=(12,8*img.shape[0]/img.shape[1]))
    ax[0,0].imshow(omega*img, cmap='gray')
    ax[0,0].set_title('observed')

    ax[0,1].imshow(img_hat, cmap='gray')
    ax[0,1].set_title('recovered')

    ax[0,2].imshow(img, cmap='gray')
    ax[0,2].set_title('original')


    sample_rate = 0.1
    number_elements = int(sample_rate*img.shape[0]*img.shape[1])
    omega = make_support(img.shape, number_elements)


    img_hat, rel_error, time = algo(img, omega, eadmm, None, 0.3)
    print(rel_error[-1], time)
    img_hat, rel_error, time = algo(img, omega, eadmm, 'hb', 0.3)
    print(rel_error[-1], time)

    ax[1,0].imshow(omega*img, cmap='gray')

    ax[1,1].imshow(img_hat, cmap='gray')

    for i in range(2):
        for j in range(3):
            ax[i,j].axis('off')

    plt.subplots_adjust(wspace=0.01, hspace=0.02)

    fig.savefig("image_recovery.pdf", bbox_inches="tight")


if __name__ == '__main__':
    #convergence(0.1)
    make_plot_convergence()
    
