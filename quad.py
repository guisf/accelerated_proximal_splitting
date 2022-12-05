""" 
Simulate the optimization algorithms on a quadratic problem
and compare to the exact solution.
We plot the trajectories.

"""

import numpy as np
from scipy.special import gamma as Gamma
from scipy.special import jv as Bessel


# omega term for the three functions
om1 = 1./2.
om2 = 1./3.
om3 = 1./5.

# initial point
x0 = 10


class Quadratic:

    def __init__(self, method='admm', stepsize=1, time=100, accel='', damp=3):
        self.method = method  # algorithm, admm or davis-yin
        self.stepsize = stepsize # stepsize
        self.accel = accel # type of acceleration, constant or decaying 
        self.damp = damp # damping coefficient
        self.maxiter = int(time/np.sqrt(stepsize))
        self.scores_ = []
        self.trajectory_ = []
        self.time_ = []

    def fit(self):
        
        # controls each of the quadratic functions
        self.omega1 = om1
        self.omega2 = om2
        self.omega3 = om3

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
        w2 = self.omega2
        w3 = self.omega3
        Q1 = 1./(h*(w1**2)+1)
        Q2 = 1./(h*(w2**2)+1)
        for k in range(1, self.maxiter):
            self.k = k
            xold = x

            x12 = Q1*(xhat - h*(w3**2)*xhat + h*c)
            x = Q2*(x12 - h*c)
            c = c + (1./h)*(x - x12)
            xhat = self._accelerate(x, xold)
           
            self.x = x
            self.scores_.append(self._objective(x))
            self.trajectory_.append(x)
            self.time_.append(k*np.sqrt(h))
    
    def _davisyin(self):
        
        x = self.x
        xhat = x
        
        h = self.stepsize
        w1 = self.omega1
        w2 = self.omega2
        w3 = self.omega3
        Q1 = 1./(h*(w1**2)+1)
        Q2 = 1./(h*(w2**2)+1)
        for k in range(1, self.maxiter):
            self.k = k
            xold = x

            x14 = Q1*(xhat)
            x12 = 2*x14 - xhat
            x34 = Q2*(x12 - h*(w3**2)*x14)
            x = xhat + x34 - x14
            xhat = self._accelerate(x, xold)
            
            self.scores_.append(self._objective(x))
            self.trajectory_.append(x)
            self.time_.append(k*np.sqrt(h))
    
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
        return 0.5*(self.omega1**2+self.omega2**2+self.omega3**2)*(x**2)

def exact_sol_const(x0, damp, h, T):
    max_iter = int(T/h)
    om_square = om1**2 + om2**2 + om3**2
    xi = np.sqrt(4*om_square-damp**2)
    trajectory = [x0]
    time = [0]
    for k in range(1, max_iter):
        t = h*k
        x = x0*np.exp(-damp*t/2)*(np.cos(xi*t/2) + (damp/xi)*np.sin(xi*t/2))
        trajectory.append(x)
        time.append(t)
    return time, trajectory

def exact_sol_decaying(x0, r, h, T):
    max_iter = int(T/h)
    om_square = om1**2 + om2**2 + om3**2
    w = np.sqrt(om_square)
    nu = (r-1)/2.
    trajectory = [x0]
    time = [0]
    for k in range(1, max_iter):
        t = h*k
        x = x0*np.power(2, nu)*np.power(w, -nu)*Gamma(nu+1)*np.power(t, -nu)*\
            Bessel(nu, w*t)
        trajectory.append(x)
        time.append(t)
    return time, trajectory


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    T = 25   # simulation time
    eta = 0.2 # damping for constant case
    r = 3 # damping for decaying case

    # constant damping case
    time, exact_traj = exact_sol_const(x0, eta, 0.01, T)
    q_admm = Quadratic('admm',0.0001,T,'constant',eta)
    q_admm.fit()
    q_dy = Quadratic('davis-yin',0.0001,T,'constant',eta)
    q_dy.fit()
   
    # decaying damping case
    time2, exact_traj2 = exact_sol_decaying(x0, r, 0.01, T)
    q_admm2 = Quadratic('admm',0.0001,T,'decaying',r)
    q_admm2.fit()
    q_dy2 = Quadratic('davis-yin',0.0001,T,'decaying',r)
    q_dy2.fit()

    #fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.2, 5))
    fig, axs = plt.subplots(2, 1, sharex=True)
    ax1, ax2 = axs
    ax1.plot(time, exact_traj, '-', color='black', label='exact', linewidth=3)
    ax1.plot(q_admm.time_, q_admm.trajectory_, 
            's', fillstyle='left', markevery=0.05,
            color='tab:blue',
            markeredgecolor='k', markeredgewidth=0.5,
            markersize=10,
            label='ADMM constant', alpha=0.7)
    ax1.plot(q_dy.time_, q_dy.trajectory_, 
            's', fillstyle='right', markevery=0.05, 
            color='tab:orange',
            markeredgecolor='k', markeredgewidth=0.5,
            markersize=10,
            label='DY constant', alpha=0.7)
    ax1.legend(loc=1, ncol=3, handletextpad=0.3, columnspacing=1,
                handlelength=1.5)
    #ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$x(t)$')
    #fig.savefig('quad_traj_const.pdf', bbox_inches='tight')
    
    ax2.plot(time2, exact_traj2, '-', color='black', label='exact', 
                linewidth=3)
    ax2.plot(q_admm2.time_, q_admm2.trajectory_, 
            's', fillstyle='left', markevery=0.05,
            color='tab:blue',
            markeredgecolor='k', markeredgewidth=0.5,
            markersize=10,
            label='ADMM decaying', alpha=0.7)
    ax2.plot(q_dy2.time_, q_dy2.trajectory_, 
            's', fillstyle='right', markevery=0.05, 
            color='tab:orange',
            markeredgecolor='k', markeredgewidth=0.5,
            markersize=10,
            label='DY decaying', alpha=0.7)
    #ax2.legend(loc=0)
    ax2.legend(loc=1, ncol=3, handletextpad=0.3, columnspacing=1, 
                handlelength=1.5)
    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'$x(t)$')

    plt.subplots_adjust(wspace=0,hspace=0)
    
    fig.savefig('quad_traj.pdf', bbox_inches='tight')

