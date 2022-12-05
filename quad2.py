"""
Simulate the optimization algorithms on a quadratic problem
and compare to the exact solution
here we plot the global error as a function of the step size
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
        self.trajectory_ = np.array(self.trajectory_)
    
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
        self.trajectory_ = np.array(self.trajectory_)
    
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
    return time, np.array(trajectory)

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
    import pickle

    T = 25    # simulation time
    eta = 0.2 # damping coefficient
    r = 3
    
    h_range = [1/(4**i) for i in range(10)]
    h_range.reverse()
    #print(h_range)
    
    # generate data
    """
    scores1 = []
    scores2 = []
    for h in h_range:
        time, exact_traj = exact_sol_const(x0, eta, h, T)
        admm = Quadratic('admm',h**2,T,'constant',eta)
        admm.fit()
        dy = Quadratic('davis-yin',h**2,T,'constant',eta)
        dy.fit()
        traj1 = admm.trajectory_
        traj2 = dy.trajectory_
        scores1.append(np.max(np.abs(exact_traj - traj1)))
        scores2.append(np.max(np.abs(exact_traj - traj2)))
    
    pickle.dump([scores1,scores2], open('order.pickle', 'wb'))
    """

    # after generating data, comment above and load for plotting
    scores1, scores2 = pickle.load(open('order.pickle', 'rb'))

    #fig, axs = plt.subplots(1, 2, sharex=True, figsize=(7.2, 5))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ref_line = [7*h for h in h_range]
    ax.plot(h_range[1:], ref_line[1:], '--', color='black', 
            label=r'theoretical $\mathcal{O}(h)$')
    ax.plot(h_range[1:], scores1[1:], 's', fillstyle='left', 
            color='tab:blue',
            markeredgecolor='k', 
            markeredgewidth=0.5, markersize=12,
            label='ADMM')
    ax.plot(h_range[1:], scores2[1:], 's', fillstyle='right', 
            color='tab:orange',
            markeredgecolor='k', 
            markeredgewidth=0.5, markersize=12,
            label='DY')
    ax.legend(loc=0)
    ax.set_xlabel(r'step size`')
    ax.set_ylabel(r'max error')
    fig.savefig('quad_const_order.pdf', bbox_inches='tight')
    
