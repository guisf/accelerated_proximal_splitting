"""Instance of the LASSO regression problem."""

import numpy as np
from scipy import stats
from scipy import sparse

from prettytable import PrettyTable

from lasso import LASSO

import matplotlib.pyplot as plt
from matplotlib import rc


def generate_data(m, n, density=0.05, sigma=0.001, verbose=False):
    """Generate data to the problem. 'density' controls the level
    of sparsity in the ground truth and 'sigma' the gaussian noise.
    
    """
    rvs = stats.norm().rvs
    x_true = sparse.random(1, n, density, data_rvs=rvs)
    x_true = x_true.toarray()[0]
    A = np.random.normal(size=(m, n))
    A = A/np.sqrt((A**2).sum(axis=0))
    v = np.sqrt(sigma)*np.random.normal(size=m)
    b = A.dot(x_true) + v
    if verbose:
        non_zero = len(x_true[np.where(x_true!=0)])
        signal_noise = (np.linalg.norm(A.dot(x_true))**2)/(np.linalg.norm(v)**2)
        print("LASSO with %i examples, %i variables"%(m, n))
        print("non_zero=%i, signal/noise ratio: %.2f"%(non_zero, signal_noise))
    return x_true, A, b, v

def single_test(params, m=500, n=2500, density=0.05, sigma=1e-3,
                maxiter=1000, tol=1e-10):
    """Testing a single problem with different optimizers.
    'params' contains the options for all the solvers.
    
    """
    x_true, A, b, v = generate_data(m, n, density, sigma, verbose=True)
    lamb = 0.1*np.linalg.norm(A.T.dot(b), ord=np.inf)
    error = lambda x: np.linalg.norm(x-x_true)/np.linalg.norm(x_true)
    table = PrettyTable()
    table.field_names = ['Algo','Rel Error', 'Objective', 'Iter']
    for algo, p in params.items():
        l = LASSO(maxiter=maxiter, tol=tol, lamb=lamb, **p)
        l.fit(A, b)
        table.add_row([algo, error(l.coef_), l.score_, l.k])
    print(table)

def run_many(params, m=500, n=2500, density=0.05, sigma=1e-3, 
             num_exp=10, maxiter=150, tol=1e-10):
    """Generate convergence plot."""
    
    results = {algo: [] for algo in params.keys()}
    for _ in range(num_exp):
        x_true, A, b, v = generate_data(m, n, density, sigma, verbose=True)
        lamb = 0.1*np.linalg.norm(A.T.dot(b), ord=np.inf)
        
        l = LASSO(lamb=lamb,method='cvx',tol=tol*1e-3)
        l.fit(A, b)
        fstar = l.score_
        
        for algo, p in params.items():
            l = LASSO(lamb=lamb, maxiter=maxiter, tol=tol, **p)
            l.fit(A, b)
            results[algo].append([np.abs(f - fstar)/fstar for f in l.scores_])
    
    return results

def plot_many(results, style, ax, maxiter=np.inf):
    """Plot the results from the above function."""
    for algo, result in results.items():
        if 'ADMM2' in algo:
            continue
        m = np.min([len(line) for line in result])
        r = np.array([line[:m] for line in result])
        means = r.mean(axis=0)
        std = r.std(axis=0)
        m = int(np.min([maxiter, len(means)]))
        means = means[:m]
        std= std[:m]
        algo_legend = legend_name.get(algo, algo)
        ax.errorbar(range(len(means)),means,yerr=std,label=algo_legend,
                    **style[algo])


if __name__ == '__main__':

    import pickle

    #fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8.0,4))
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8.0,4))
    ax1, ax2 = axs
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    legend_name = {
        'FBS': 'FB',
        'FBS decaying': 'FB decaying',
        'FBS constant': 'FB constant',
        'Tseng decaying': 'Tseng decaying',
        'Tseng constant': 'Tseng constant',
        'DR decaying': 'DR decaying',
        'DR constant': 'DR constant',
        'ADMM decaying': 'ADMM decaying',
        'ADMM constant': 'ADMM constant',
    }

    mke = 5; ee = 5;
    style = {
    'FBS': {'linestyle':'-','color':'tab:blue','marker':'None',
            #'markeredgecolor':'black',
            'elinewidth':2,
            'markevery':mke,'errorevery':ee,'fillstyle':'none'},
    'FBS decaying': {'linestyle':'-','color':'tab:green','marker':'None',
            #'markeredgecolor':'black',
            'elinewidth':2,
            'markevery':mke,'errorevery':ee,'fillstyle':'none'},
    'FBS constant': {'linestyle':'-','color':'tab:purple','marker':'None',
            #'markeredgecolor':'black',
            'elinewidth':2,
            'markevery':mke,'errorevery':ee,'fillstyle':'none'},

    'Tseng': {'linestyle':'-','color':'tab:orange','marker':'None',
            #'markeredgecolor':'black',
            'elinewidth':2,
            'markevery':mke,'errorevery':ee,'fillstyle':'none'},
    'Tseng decaying': {'linestyle':'-','color':'tab:red','marker':'None',
            #'markeredgecolor':'black',
            'elinewidth':2,
            'markevery':mke,'errorevery':ee,'fillstyle':'none'},
    'Tseng constant': {'linestyle':'-','color':'tab:brown','marker':'None',
            #'markeredgecolor':'black',
            'elinewidth':2,
            'markevery':mke,'errorevery':ee,'fillstyle':'none'},
    
    'DR': {'linestyle':'-','color':'tab:pink','marker':'None',
            #'markeredgecolor':'black',
            'elinewidth':2,
            'markevery':mke,'errorevery':ee,'fillstyle':'none'},
    'DR decaying': {'linestyle':'-','color':'tab:olive','marker':'None',
            #'markeredgecolor':'black',
            'elinewidth':2,
            'markevery':mke,'errorevery':ee,'fillstyle':'none'},
    'DR constant': {'linestyle':'-','color':'mediumslateblue','marker':'None',
            #'markeredgecolor':'black',
            'elinewidth':2,
            'markevery':mke,'errorevery':ee,'fillstyle':'none'},
    
    'ADMM': {'linestyle':'-','color':'tab:gray','marker':'None',
            #'markeredgecolor':'black',
            'elinewidth':2,
            'markevery':mke,'errorevery':ee,'fillstyle':'none'},
    'ADMM decaying': {'linestyle':'-','color':'tab:cyan','marker':'None',
            #'markeredgecolor':'black',
            'elinewidth':2,
            'markevery':mke,'errorevery':ee,'fillstyle':'none'},
    'ADMM constant': {'linestyle':'-','color':'coral','marker':'None',
            #'markeredgecolor':'black',
            'elinewidth':2,
            'markevery':mke,'errorevery':ee,'fillstyle':'none'},
    }

    h = 0.08
    params = {
    'FBS':{'method':'fbs','stepsize':h,'accel':''},
    'FBS decaying':{'method':'fbs','stepsize':h,'accel':'decaying','damp':3},
    'FBS constant':{'method':'fbs','stepsize':h,'accel':'constant','damp':0.5},
    'Tseng': {'method':'tseng','stepsize':h,'accel':''},
    'Tseng decaying':{'method':'tseng','stepsize':h,'accel':'decaying',
                        'damp':3},
    'Tseng constant': {'method':'tseng','stepsize':h,'accel':'constant',
                        'damp':0.5}
    }
    #results = run_many(params, num_exp=10, maxiter=300, tol=1e-10)
    #pickle.dump(results, open('res1_v2.pickle', 'wb'))
    results = pickle.load(open('res1.pickle', 'rb'))
    plot_many(results, style, ax1, maxiter=110)

    h = 0.08
    params = {
    'DR':{'method':'drs','stepsize':h,'accel':''},
    'DR decaying': {'method':'drs','stepsize':h,'accel':'decaying','damp':3},
    'DR constant': {'method':'drs','stepsize':h,'accel':'constant','damp':0.5},
    'ADMM': {'method':'admm','stepsize':h,'accel':''},
    'ADMM decaying':{'method':'admm','stepsize':h,'accel':'decaying','damp':3},
    'ADMM constant':{'method':'admm','stepsize':h,'accel':'constant',
                        'damp':0.5},
    }
    #results = run_many(params, num_exp=10, maxiter=300, tol=1e-10)
    #pickle.dump(results, open('res2_v2.pickle', 'wb'))
    results = pickle.load(open('res2.pickle', 'rb'))
    plot_many(results, style, ax2, maxiter=110)

    ax1.set_ylabel('relative error')
    #ax2.set_ylabel('relative error')
    ax1.set_xlabel('iteration')
    ax2.set_xlabel('iteration')
    ax1.legend()
    ax2.legend()
    plt.subplots_adjust(wspace=0,hspace=0)

    #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #fig.legend(lines, labels, ncol=3, bbox_to_anchor=(0.88,1.3), 
    #            columnspacing=1.0, handletextpad=0.4, fontsize='small')

    fig.savefig('lasso.pdf', bbox_inches='tight')

