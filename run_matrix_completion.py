
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matrix_completion 


def random_data_experiment(num_experiments=20, n=100, r=5, samp_ratio=0.3,
        maxiter=3000, tol=1e-10, mu=3.5, 
        damp_coef = {'decaying': 3, 'constant': 0.1, '': 3}, 
        annealing=False):
    """Generate data from several runs."""
    results = {}
    num_iter = {}
    exp = 0
    while exp < num_experiments:
        A = np.random.normal(loc=3, scale=1, size=(n,r))
        B = np.random.normal(loc=3, scale=1, size=(n,r))
        M = A.dot(B.T)
        num_obs = int(np.prod(M.shape)*samp_ratio)
        Mask = matrix_completion.make_support(M.shape, num_obs)
        Mobs = Mask*M
        sigma = Mobs.std()
        a = Mobs[np.where(Mobs != 0)].min()-0.5*sigma
        b = Mobs.max()+0.5*sigma

        if annealing:
            mus = matrix_completion.mu_schedulle(Mobs)

        mc = matrix_completion.MatrixCompletion(mu=mu, lower=a, higher=b,
                                                M=M, Mobs=Mobs, Mask=Mask)
        for algo in ['dy', 'admm', 'admm2']:
            for accel in ['', 'decaying', 'constant']:
                damp = damp_coef[accel]
                if annealing:
                    mc.solve_annealing(mus, method=algo, stepsize=1,
                        accel=accel, damp=damp, maxiter=maxiter, tol=tol)
                    if mc.k >= 3000:
                        continue
                else:
                    mc.solve(method=algo, stepsize=1, accel=accel, damp=damp,
                         maxiter=maxiter, tol=tol)
                key = '%s %s'%(algo.upper(), accel)
                print(exp, key, mc.total_error(), mc.rank_, mc.k)
                if key not in results:
                    results[key] = [mc.total_error_history()]
                    num_iter[key] = [mc.k]
                else:
                    results[key].append(mc.total_error_history())
                    num_iter[key].append(mc.k)
        exp += 1

    return results, num_iter

def plot_convergence(results, output='matrix_completion1.pdf', 
                        cut=2000, mke=10):
    """Plot relative error per iteration."""
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                    figsize=(8, 4))
    #fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    #markers = ['o', 's', 'D', '^', 'p', 'h']
    markers = [None, None, None, None, None, None]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                'tab:brown']
    lname = ['DY', 'DY decaying', 'DY constant',
             'ADMM', 'ADMM decaying', 'ADMM constant']
    i = 0
    for algo, ax in zip(['dy', 'admm'], axes.flat):
        ax.set_yscale('log')
        for accel in ['', 'decaying', 'constant']:
            key = '%s %s'%(algo.upper(), accel)
            data = results[key]
            # trim because different runs could have different iterations
            num_iter = np.min([len(experiment) for experiment in data])
            num_iter = np.min([num_iter, cut])
            data = np.array([experiment[:num_iter] for experiment in data])
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            ax.errorbar(range(len(mean)), mean, yerr=std, 
                        label=lname[i], errorevery=mke, color=colors[i], 
                        marker=markers[i], markevery=mke, markeredgecolor='k',
                        elinewidth=2, ecolor=colors[i])
            i += 1
        #ax.legend()
    axes[0].set_xlabel(r'iteration')
    axes[1].set_xlabel(r'iteration')
    axes[0].set_ylabel(r'relative error')
    fig.subplots_adjust(hspace=0, wspace=0) 
    #axes[0].legend(loc=3)
    #axes[1].legend(loc=3)
    axes[0].legend(loc=0)
    axes[1].legend(loc=0)
    #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #fig.legend(lines, labels, ncol=2, bbox_to_anchor=(0.8,1.23),
    #            columnspacing=5.0, handletextpad=0.4, 
    #            fontsize='small')
    
    fig.savefig(output, bbox_inches='tight')

def plot_iteration(num_iter, output='matrix_completion_box1.pdf'):
    """Boxplot of iteration until convergence."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df = pd.DataFrame(num_iter)
    df = df.drop(['ADMM2 ', 'ADMM2 decaying', 'ADMM2 constant'], axis=1)
    df = df.rename(columns={
            'DY ': 'DY', 
            'DY decaying': 'DY decaying',
            'DY constant': 'DY constant',
            'ADMM ': 'ADMM', 
            'ADMM decaying': 'ADMM decaying',
            'ADMM constant': 'ADMM constant'}
    )
    #ax.boxplot(num_iter.values(), labels=num_iter.keys())
    #colors = {'dy': 'lightskyblue', 'admm': 'moccasin', 'admm2': 'lightgreen'}
    colors = {'dy': 'tab:blue', 'admm': 'tab:orange'}
    my_pal = {}
    for algo in ['dy', 'admm']:
        for accel in ['', 'decaying', 'constant']:
            key = '%s %s'%(algo.upper(), accel)
            key = key.rstrip('-')
            my_pal[key] = colors[algo]
    #sns.boxplot(data=df, ax=ax, palette=my_pal, linewidth=1, saturation=1)
    sns.boxplot(data=df, ax=ax, linewidth=2)
    plt.xticks(rotation=30)
    #for tick in ax.xaxis.get_major_ticks():
    #    tick.label.set_fontsize(14)
    ax.set_ylabel(r'iterations')
    #ax.set_ylim([0,5000])
    fig.savefig(output, bbox_inches='tight')


##############################################################################
if __name__ == "__main__":

    import sys

    algo_names = {
                  'DY ': 'DY', 
                  'DY decaying': 'DY-decaying',
                  'DY constant': 'DY-constant',
                  'E-ADMM ': 'ADMM', 
                  'E-ADMM decaying': 'ADMM-decaying',
                  'E-ADMM constant': 'ADMM-constant',
                  }

    #results, num_iter = random_data_experiment(num_experiments=10, 
    #                        samp_ratio=0.4)
    #pickle.dump([results, num_iter], open('mat1.pickle', 'wb'))
    
    results, num_iter = pickle.load(open('mat1.pickle', 'rb'))
    
    # convergence plot
    plot_convergence(results, cut=100, output='mc_convergence1.pdf', mke=5)
    # all methods converge to relative error of 5e-3
    plot_iteration(num_iter, output='mc_iteration1.pdf')

    #results, num_iter = random_data_experiment(num_experiments=10,
    #                        samp_ratio=0.4, annealing=True,
    #                    damp_coef = {'decaying': 3, 'constant': 0.5, '': 3})
    #pickle.dump([results, num_iter], open('mat2.pickle', 'wb'))
    results, num_iter = pickle.load(open('mat2.pickle', 'rb'))
    # convergence plot
    plot_convergence(results, output='mc_convergence2.pdf', cut=1500, mke=60)
    # all solutions converge to 1e-10
    plot_iteration(num_iter, output='mc_iteration2.pdf')
