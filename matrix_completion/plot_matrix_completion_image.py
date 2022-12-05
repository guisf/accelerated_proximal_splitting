# plot results for matrix completion using an image
# the data was generated from the jupyter notebook: "some_experiments.ipynb"

import matplotlib.pyplot as plt
import pickle


def plot_single_run():
    
    # loading data for recovery *without annealing*
    # this was generated in the jupyter notebook: "some_experiments.ipynb"
    # on the "hard" problem
    with open('data/hard_dy_single_results.pickle', 'rb') as f:
        err1, err2, err3, rank1, rank2, rank3 = pickle.load(f)
    with open('data/hard_eadmm_single_results.pickle', 'rb') as f:
        err4, err5, err6, rank4, rank5, rank6 = pickle.load(f)
    
    fig = plt.figure(figsize=(8.0,4.0))
    
    ax1 = fig.add_subplot(121)
    #ax1.set_yscale('log')
    mi = 70
    #ax1.plot(err1[:mi],'--',marker='o',fillstyle='none',markevery=0.1,
    #         markeredgecolor='k',color=(0.15,0.43,0.68),label='DY')
    ax1.plot(range(1,mi+1),err1[:mi],'-',color='tab:blue',
             marker=None,markevery=0.05,markeredgecolor='k',label='DY')
    ax1.plot(range(1,mi+1),err2[:mi],'-',color='tab:orange',
             label='DY decaying',
             marker=None,markevery=0.05,markeredgecolor='k')
    ax1.plot(range(1,mi+1),err3[:mi],'-',color='tab:green',
             label='DY constant',
             marker=None,markevery=0.05,markeredgecolor='k')
    
    ax2 = fig.add_subplot(122, sharey=ax1)
    #ax2.set_yscale('log')
    ax2.plot(range(1,mi+1),err4[:mi],'-',color='tab:red',label='ADMM',
             marker=None,markevery=0.05,markeredgecolor='k')
    ax2.plot(range(1,mi+1),err5[:mi],'-',color='tab:purple',
             label='ADMM decaying',
             marker=None,markevery=0.05,markeredgecolor='k')
    ax2.plot(range(1,mi+1),err6[:mi],'-',color='tab:brown',
             label='ADMM constant',
             marker=None,markevery=0.05,markeredgecolor='k')
    
    plt.subplots_adjust(wspace=0,hspace=0)
    ax1.legend()
    ax2.legend()
    ax2.tick_params(labelleft=False)
    ax1.set_ylabel('relative error')
    ax1.set_xlabel('iteration')
    ax2.set_xlabel('iteration')
    
    fig.savefig('boat_singlerun.pdf', bbox_inches='tight')

    mi = 70
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(range(1,mi+1),rank4[:mi],'-',color='tab:red',label='ADMM',
             marker=None,markevery=0.02,markeredgecolor='k')
    ax.plot(range(1,mi+1),rank5[:mi],'-',color='tab:purple',
             label='ADMM decaying',
             marker=None,markevery=0.02,markeredgecolor='k')
    ax.plot(range(1,mi+1),rank6[:mi],'-',color='tab:brown',
             label='ADMM constant',
             marker=None,markevery=0.02,markeredgecolor='k')
    ax.set_ylabel('rank')
    ax.set_xlabel('iteration')
    #ax.legend(loc='center right')
    ax.legend()
    ax.set_yticks([70,90,110,130,150])
    fig.savefig('boat_single_rank.pdf', bbox_inches='tight')


def plot_annealing_run():
    
    # this is *with annealing*
    # outcome of Davis-Yin with non acceleration: 
    # 0.0035215859789830698 150 766.851968050003  
    with open('data/hard_dy_annealing_noaccell.pickle', 'rb') as f:
        harderrors1, hardranks1 = pickle.load(f)
    # davis-yin with nesterov acceleration
    # 0.00015660277383090665 71 684.9864530563354
    with open('data/hard_dy_annealing_nest.pickle', 'rb') as f:
        harderrors2, hardranks2 = pickle.load(f)
    # davis-yin with heavy ball acceleration
    # 0.0005343848669413402 77 535.7635216712952
    with open('data/hard_dy_annealing_hb.pickle', 'rb') as f:
        harderrors3, hardranks3 = pickle.load(f)
    # outcome of ADMM with non acceleration: 
    # 0.00352171282036421 150 797.9736518859863
    with open('data/hard_admm_annealing_noaccell.pickle', 'rb') as f:
        harderrors4, hardranks4 = pickle.load(f)
    # ADMM with nesterov acceleration
    # 0.00015668171037163842 71 697.3146719932556
    with open('data/hard_admm_annealing_nest.pickle', 'rb') as f:
        harderrors5, hardranks5 = pickle.load(f)
    # ADMM with heavy ball acceleration
    # 0.0005344513100527767 77 509.06831097602844
    with open('data/hard_admm_annealing_hb.pickle', 'rb') as f:
        harderrors6, hardranks6 = pickle.load(f)
    
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    ax1.set_yscale('log')
    ax1.plot(range(1,len(harderrors1)+1),harderrors1,'-',color='tab:blue',
             label='DY',
             marker=None,markevery=0.05,markeredgecolor='k')
    ax1.plot(range(1,1970+1),harderrors2[:1970],'-',color='tab:orange',
             label='DY decaying',
             marker=None,markevery=0.05,markeredgecolor='k')
    ax1.plot(range(1,len(harderrors3)+1),harderrors3,'-',color='tab:green',
             label='DY constant',
             marker=None,markevery=0.05,markeredgecolor='k')
    
    ax2 = fig.add_subplot(122, sharey=ax1)
    ax2.set_yscale('log')
    ax2.plot(range(1,len(harderrors4)+1),harderrors4,'-',color='tab:red',
             label='ADMM',
             marker=None,markevery=0.05,markeredgecolor='k')
    ax2.plot(range(1,1970+1),harderrors5[:1970],'-',color='tab:purple',
             label='ADMM decaying',
             marker=None,markevery=0.05,markeredgecolor='k')
    ax2.plot(range(1,len(harderrors6)+1),harderrors6,'-',color='tab:brown',
             label='ADMM constant',
             marker=None,markevery=0.05,markeredgecolor='k')

    ax1.set_ylabel('relative error')
    ax1.set_xlabel('iteration')
    ax2.set_xlabel('iteration')
    ax2.tick_params(labelleft=False)
    ax1.legend()
    ax2.legend()

    plt.subplots_adjust(hspace=0,wspace=0)

    fig.savefig('boat_annealing.pdf', bbox_inches='tight')

def plot_boats():
    # write one of the recovered matrices
    with open('data/im_original_observed.pickl', 'rb') as f:
        M, Mobs2 = pickle.load(f)
    with open('data/hard_admm_hb_single_reco.pickle', 'rb') as f:
        Xhat_single, = pickle.load(f)
    with open('data/hard_admm_nest_reco.pickle', 'rb') as f:
        Xhat_aneal, = pickle.load(f)

    fig, ax =plt.subplots(nrows=2,ncols=2, figsize=(10, 9.5))

    # size: 974 x 1194 pixels, rank = 70
    ax[0,0].imshow(M, cmap='gray')
    ax[0,0].axis('off')
    ax[0,0].set_title(r'original', loc='right', pad=8)

    # samping ratio 0.3
    ax[0,1].imshow(Mobs2, cmap='gray')
    ax[0,1].set_title(r'observed', loc='right', pad=8)
    ax[0,1].axis('off')

    # ($E = 7.9\times 10^{-2}}, \, r=70$)')
    ax[1,0].imshow(Xhat_single, cmap='gray')
    ax[1,0].set_title(r'recovered (single)', loc='right', pad=8)
    ax[1,0].axis('off')

    # ($E = 1.6\times 10^{-4}}, \, r=71$)')
    ax[1,1].imshow(Xhat_aneal, cmap='gray')
    ax[1,1].set_title(r'recovered (annealing)', loc='right', pad=8)
    ax[1,1].axis('off')

    plt.subplots_adjust(wspace=0.03, hspace=0.0)

    fig.savefig('boats.pdf', bbox_inches='tight')


if __name__ == '__main__':
    
    plot_single_run()
    plot_annealing_run()
    #plot_boats()
