"""Matrix completion problem with a positive random matrix

This is the example from

An Alternating Direction Algorithm for Matrix Completion with Nonnegative 
Factors, Xu et. al. 2011

Here we just plot the results

"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys

#mpl.style.use('seaborn-bright')


n = 500
ranks = range(20, 52, 2)
observed_rates = [0.75, 0.50, 0.25]
#markers = ['o', 'h', 's']
markers = [None, None, None]
ylabel = r'relative error ($\times 10^{-2}$)'
#ylabel = 'time (s)'
output = 'positive_rel_error.pdf'
#output = 'positive_time.pdf'
column = 'obs_error'
#column = 'time'


df = pd.read_csv('data/pos_mat_compl_50trials.csv', 
                 names=['algorithm', 'rank', 'sample_rate',
                        'time', 'obs_error', 'iterations', 'true_error'])

def make_plot(ax, df, observed_rates, alg_name, column, title):

    #ax.set_yscale('log')

    for marker, sample_rate in zip(markers, observed_rates):

        data = df.loc[(df['sample_rate']==sample_rate) & \
                  (df['algorithm']==alg_name)]
        #print(alg_name, len(data.loc[data['time'] > 8]))
        #data = data.loc[df['time'] < 7]
        alg_means = [data.loc[data['rank']==r][column].mean() 
                        for r in ranks]
        alg_stds = [data.loc[data['rank']==r][column].std() 
                        for r in ranks]
        ax.errorbar(ranks, alg_means, yerr=alg_stds,
                label=r'$\kappa_{\textnormal{obs}}=%.2f$'%(sample_rate),
                marker=marker, elinewidth=1)
        #ax.plot(ranks, alg_means,
        #        label=r'$\kappa_{\textnormal{obs}}=%.2f$'%(sample_rate))
        
        ax.set_title(title)
        ax.set_xlim([20, 50])


#fig, axs = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(6,6))
fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4,4))

make_plot(axs[0,0], df, observed_rates, 'DY', column, 'DY')
make_plot(axs[0,1], df, observed_rates, 'E2-ADMM', column, 'E2-ADMM')
make_plot(axs[0,2], df, observed_rates, 'E-ADMM', column, 'E-ADMM')

make_plot(axs[1,0], df, observed_rates, 'DY-Nest', column, 'DY (Nest)')
make_plot(axs[1,1], df, observed_rates, 'E2-ADMM-Nest', column, 'E2-ADMM (Nest)')
make_plot(axs[1,2], df, observed_rates, 'E-ADMM-Nest', column, 'E-ADMM (Nest)')

make_plot(axs[2,0], df, observed_rates, 'DY-HB', column, 'DY (HB)')
make_plot(axs[2,1], df, observed_rates, 'E2-ADMM-HB', column, 'E2-ADMM (HB)')
make_plot(axs[2,2], df, observed_rates, 'E-ADMM-HB', column, 'E-ADMM (HB)')

handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc=(0.1, 0), ncol=3)
axs[1,0].set_ylabel(ylabel)
#axs[1,0].set_yticks([0,2,4,6])
#axs[1,0].set_yticklabels(['0.00','2.00', '4.00', '6.00'])
#axs[1,0].set_ylim([0,6])

axs[1,0].set_ylim([0,0.02])
axs[1,0].set_yticks([0.0, 0.01, 0.02])
axs[1,0].set_yticklabels(['0', '1', '2'])

axs[2,1].set_xlabel('rank')
axs[2,1].set_xticks([20,30,40,50])
plt.subplots_adjust(hspace=0.3)

fig.savefig(output, bbox_inches='tight')

