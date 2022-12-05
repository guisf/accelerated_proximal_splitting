"""Simulate the stochastic optimization algorithms on a quadratic problem
and compare to the exact solution of the Fokker-Planck equation
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.stats
import sys


class Quadratic:

    def __init__(self, method='admm', stepsize=1, time=100, accel='', damp=3,
                noise_type='prox'):
        self.method = method  # algorithm, admm or davis-yin
        self.stepsize = stepsize # stepsize
        self.accel = accel # type of acceleration, constant or decaying 
        self.damp = damp # damping coefficient
        self.noise_type = noise_type
        if accel=='constant' or accel=='decaying':
            self.maxiter = int(time/np.sqrt(stepsize))
        else:
            self.maxiter = int(time/stepsize)
        self.scores_ = []
        self.trajectory_ = []
        self.time_ = []

    def  _minibatch(self):
        return np.random.choice(self.omegas, size=self.batch_size, 
                                replace=False)
    
    def fit(self, x0, om1, om2, omegas, batch_size):
        
        self.omega1 = om1
        self.omega2 = om2
        self.omegas = omegas
        self.batch_size = batch_size
        self.N = len(self.omegas)

        # initialize
        self.x = x0
        self.trajectory_.append(x0)
        self.time_.append(0)
        self.scores_.append(self._objective(self.x))

        if self.method == 'admm':
            self._admm()
        elif self.method == 'davis-yin':
            self._davisyin()
        else:
            raise ValueError('No optimization method specified.')

    def _admm(self):
        
        x = self.x
        xhat = x
        c = 0
        
        h = self.stepsize
        w1 = self.omega1
        Q1 = 1./(h*(w1**2)+1)
        for k in range(1, self.maxiter):
            self.k = k
            xold = x

            oms = self._minibatch()
            if self.noise_type == 'prox':
                w2_square = np.sum(oms**2)/self.batch_size
                Q2 = 1./(h*(w2_square)+1)
                w3_square = self.omega2
            else:
                w2_square = self.omega2**2
                Q2 = 1./(h*(w2_square)+1)
                w3_square = np.sum(oms**2)/self.batch_size

            x12 = Q1*(xhat - h*(w3_square)*xhat + h*c)
            x = Q2*(x12 - h*c)
            c = c + (1./h)*(x - x12)
            xhat = self._accelerate(x, xold)
           
            self.x = x
            self.scores_.append(self._objective(x))
            self.trajectory_.append(x)
            if self.accel=='constant' or self.accel=='decaying':
                self.time_.append(k*np.sqrt(h))
            else:
                self.time_.append(k*h)
    
    def _davisyin(self):
        
        x = self.x
        xhat = x
        
        h = self.stepsize
        w1 = self.omega1
        Q1 = 1./(h*(w1**2)+1)
        for k in range(1, self.maxiter):
            self.k = k
            xold = x
            
            oms = self._minibatch()
            if self.noise_type == 'prox':
                w2_square = np.sum(oms**2)/self.batch_size
                Q2 = 1./(h*(w2_square)+1)
                w3_square = self.omega2
            else:
                w2_square = self.omega2**2
                Q2 = 1./(h*(w2_square)+1)
                w3_square = np.sum(oms**2)/self.batch_size
            
            # get minibatch and compute w3**2
            oms = self._minibatch()
            w3_square = np.sum(oms**2)/self.batch_size

            x14 = Q1*(xhat)
            x12 = 2*x14 - xhat
            x34 = Q2*(x12 - h*(w3_square)*x14)
            x = xhat + x34 - x14
            xhat = self._accelerate(x, xold)
            
            self.scores_.append(self._objective(x))
            self.trajectory_.append(x)
            if self.accel == 'decaying' or self.accel == 'constant':
                self.time_.append(k*np.sqrt(h))
            else:
                self.time_.append(k*h)
    
    def _accelerate(self, x, xold):
        """Light, heavy (or no) acceleration."""
        if self.accel == 'decaying':
            y = x + (self.k)/(self.k+self.damp)*(x - xold)
        elif self.accel == 'constant':
            y = x + (1-self.damp*np.sqrt(self.stepsize))*(x - xold)
        else:
            y = x
        return y

    def _objective(self, x):
        return 0.5*(self.omega1**2+\
                    self.omega2**2+np.sum(self.omegas**2)/self.N)*(x**2)


def prob_over(lamb, D, x, t, x0):
    return np.sqrt(lamb/2/np.pi/D/(1-np.exp(-2*lamb*t)))*\
        np.exp(-lamb*(x-x0*np.exp(-lamb*t))**2/2/D/(1-np.exp(-2*lamb*t)))

def gen_data(x0, T, h, N, S, M, om1, om2, fname, **params):
    # S -> batch size, N -> number of terms
    # fname -> file name to save data (pickle)
    # params={method:'admm', stepsize: 1, time=100, accel='constant', damp:3}
    
    omegas = np.random.uniform(0.0, 1, N)
    data = []
    for i in range(M):
        q = Quadratic(**params)
        q.fit(x0, om1, om2, omegas, S)
        data.append(q.trajectory_)
    time = q.time_
    data = np.array(data)
    pickle.dump([omegas, data, time], open(fname, 'wb'))


if __name__ == '__main__':

    x0 = 10
    T = 10   # simulation time
    h = 0.1 # step size
    N = 1000 # number functions
    S = 1 # batch size
    M = 2000 # number of experiments
    om1 = 1./2.
    om2 = 1./3.

    fname = 'stoch_dy.pickle'
    #gen_data(x0, T, h, N, S, M, om1, om2,
    #        fname, method='davis-yin', stepsize=h, time=T, accel='none',
    #        noise_type='grad')

    fname2 = 'stoch_admm.pickle'
    #gen_data(x0, T, h, N, S, M, om1, om2,
    #        fname2, method='admm', stepsize=h, time=T, accel='none',
    #        noise_type='grad')

    omegas, data, time = pickle.load(open(fname, 'rb'))
    omegas2, data2, time2 = pickle.load(open(fname2, 'rb'))
    
    # plotting the histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    k = 20 
    t = time[k]
    dat = data[:,k]
    dat2 = data2[:,k]
    ax.hist(dat2, bins=20, density=True, alpha=0.8, label='ADMM')
    ax.hist(dat, bins=20, density=True, alpha=0.8, label='DY')
    ax.set_xlabel(r'$x(t)$')

    mean, std = scipy.stats.norm.fit(dat)
    mean2, std2 = scipy.stats.norm.fit(dat2)
    xmin, xmax = plt.xlim()
    xs = np.linspace(xmin, xmax, 130)
    ys = scipy.stats.norm.pdf(xs, mean, std)
    ax.plot(xs, ys, ls='--', color='black', label='Gaussian fit')
    
    ax.legend()
    #fig.savefig('stoch_hist.pdf', bbox_inches='tight')
    fig.savefig('stoch_hist.png', bbox_inches='tight')
    
    # printing out some numbers 
    lamb = om1**2+om2**2+np.sum(omegas**2)/N
    print('*** Stochastic gradient simulation ***')
    print('Parameters: N=%i, S=%i, h=%f, M=%i'%(N, S, h, M))
    print('histogram time: %f'%t)
    print('Fit DY: mean=%.4f, std=%.4f'%(mean, std))
    print('Fit ADMM: mean=%.4f, std=%.4f'%(mean2, std2))
    print('Theoretical: mean=%.4f'%(x0*np.exp(-lamb*t)))
    D = lamb/(std**2)/(1-np.exp(-2*lamb*t))
    print('Assuming D const.: D = %.4f'%D)

    # generating mean
    fig = plt.figure()
    ax = fig.add_subplot(111)
    means = []
    stds = []
    means2 = []
    stds2 = []
    means_th = []
    stds_th = []
    D = 4.
    for k in range(len(time)):
        t = time[k]
        dat = data[:,k]
        dat2 = data2[:,k]
        mean, std = scipy.stats.norm.fit(dat)
        mean2, std2 = scipy.stats.norm.fit(dat2)
        mean_th = x0*np.exp(-lamb*t)
        std_th = np.sqrt(lamb/D/(1-np.exp(-2*lamb*t)+0.00001))
        means.append(mean)
        means2.append(mean2)
        stds.append(std)
        stds2.append(std2)
        means_th.append(mean_th)
        stds_th.append(std_th)
    stds_th[0] = 0
    
    y1 = np.array(means_th) - np.array(stds_th)
    y2 = np.array(means_th) + np.array(stds_th)
    ax.fill_between(time, y1, y2, color='black', alpha=0.15)
    ax.plot(time, means_th, '--', color='black', 
        lw=2, label='Ornstein-Uhlenbeck')
    
    ax.plot(time, means2, marker='o', linestyle='none',
        markevery=3, fillstyle='left', markeredgecolor='black', 
        markeredgewidth=0.5, markersize=10,
        label='ADMM', alpha=0.7)
    ax.plot(time, means, marker='o', linestyle='none',
        markevery=3, fillstyle='right', markeredgecolor='black', 
        markeredgewidth=0.5, markersize=10,
        label='DY', alpha=0.7)
    
    ax.errorbar(time, means2, stds2, fmt='none',
        elinewidth=2, errorevery=3, ecolor='tab:red', barsabove=True,
        label=r'$\sigma$ for ADMM/DY')

    # inset plot
    axins = ax.inset_axes([0.45, 0.25, 0.5, 0.5])
    axins.set_yscale('log')
    
    y1 = np.array(means_th) - np.array(stds_th)
    y2 = np.array(means_th) + np.array(stds_th)
    axins.fill_between(time, y1, y2, color='black', alpha=0.15)
    axins.plot(time, means_th, '--', color='black', 
                lw=2)
    
    #axins.plot(time, means2, marker='s', linestyle='none',
    #    markevery=3, fillstyle='left', markeredgecolor='black', 
    #    markeredgewidth=0.5, markersize=7,
    #    label='ADMM', alpha=0.7)
    #axins.plot(time, means, marker='s', linestyle='none',
    #    markevery=3, fillstyle='right', markeredgecolor='black', 
    #    markeredgewidth=0.5, markersize=7,
    #    label='DY', alpha=0.7)
    
    axins.errorbar(time, means2, stds2, fmt='none',
        elinewidth=2, errorevery=3, ecolor='tab:red', barsabove=True)

    #axins.set_xlim(4, 10)
    #axins.set_ylim(1e-3, 1)
    #axins.set_xticklabels('')
    #axins.set_yticklabels('')
    axins.tick_params(axis='both', which='major', labelsize=8)
    axins.tick_params(axis='both', which='minor', labelsize=8)
    #ax.indicate_inset_zoom(axins, edgecolor='black')

    ax.set_ylabel(r'$\langle x(t) \rangle$')
    ax.set_xlabel(r'$t$')
    ax.legend(ncol=2, loc=9)
    
    #fig.savefig('means.pdf', bbox_inches='tight')
    fig.savefig('means.png', bbox_inches='tight')

    # checking trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for traj in data:
        ax.plot(time, traj, color='tab:blue', lw=0.3)
    for traj in data2:
        ax.plot(time2,  traj, color='tab:red', lw=0.3)
    fig.savefig('quad_stoch_traj.pdf', bbox_inches='tight')

