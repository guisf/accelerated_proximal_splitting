"""LASSO regression problem using several proximal based algorithms."""

import numpy as np

import cvxpy as cp
import cvxopt


class LASSO:
    """LASSO problem with (accelerated) proximal gradient and
    (accelerated) Tseng splitting.
    $$
        \min (1/2)\| A x - b\|^2 + \lambda \| x \|_1
    $$
    One can choose different optimizers to solve the problem.
        
    Parameters
    ----------
    lamb : [float, None]
        regularization constant; if None will be chosen automaticaly
    stepsize : float
        algorithm stepsize
    accel : string
        type of acceleration; '', 'decaying' or 'constant'
    maxiter : int
        maximum number of iterations
    method : string
        optimizer; can be 'fbs' or 'tseng'
    damp : float
        damping constant for accelerated variants
    tol: float
        tolerance on the objective function for stopping criterium

    Features
    --------
    coef_ : array
        will contain the final solution
    scores_ : array
        all values of the objective function per iteration
    score_ : float
        final value of the objective function
    
    """

    def __init__(self, lamb=None, method='fbs', stepsize=1, accel='', damp=3, 
                 maxiter=1000, tol=1e-4):
        self.lamb = lamb
        self.method = method
        self.stepsize = stepsize
        self.accel = accel
        self.damp = damp
        self.maxiter = maxiter
        self.tol = tol
        self.scores_ = []
        self.score_ = 0
        self.coef_ = 0

    def fit(self, A, b):
        """
        Call an optimizer and solve the problem.

        Parameters
        ----------
        A : 2D array
            Regressor matrix
        b : 1D array
            observed signal
        
        """
        
        self.A = A
        self.b = b
        self.AtA = A.T.dot(A)
        self.Atb = A.T.dot(b)

        if not self.lamb:
            self.lamb = 0.1*np.linalg.norm(A.T.dot(self.b), ord=np.inf)

        # initialize with 0
        self.x = np.zeros(A.shape[1])
        self.scores_.append(self._objective(self.x, self.x))

        if self.method == 'fbs':
            self._fbs()
        elif self.method == 'tseng':
            self._tseng()
        elif self.method == 'prs':
            self._peaceman_rachford()
        elif self.method == 'drs':
            self._douglas_rachford()
        elif self.method == 'admm':
            self._admm()
        elif self.method == 'admm2':
            self._admm2()
        elif self.method == 'cvx':
            self._cvx()
        else:
            raise ValueError('No optimization method specified.')

        self.coef_ = self.x
        self.score_ = self.scores_[-1]

    def _fbs(self):
        """Forward-backward splitting."""
        x = self.x
        xhat = x
        for self.k in range(1, self.maxiter):
            xold = x
            
            grad = self.AtA.dot(xhat) - self.Atb
            x = self._soft(self.stepsize*self.lamb, xhat - self.stepsize*grad)
            xhat = self._accelerate(x, xold)
            
            self.x = x
            self.scores_.append(self._objective(x, x))
            if self.k > 1 and \
                    np.abs(self.scores_[-1]-self.scores_[-2]) < self.tol:
                break

    def _tseng(self):
        """Tseng splitting."""
        x = self.x
        xhat = x
        for self.k in range(1, self.maxiter):
            xold = x

            grad = self.AtA.dot(xhat) - self.Atb
            x = self._soft(self.stepsize*self.lamb, xhat - self.stepsize*grad)
            grad2 = self.AtA.dot(x) - self.Atb
            x = x - self.stepsize*(grad2 - grad)
            xhat = self._accelerate(x, xold)
            
            self.x = x
            self.scores_.append(self._objective(x, x))
            if self.k > 1 and \
                    np.abs(self.scores_[-1]-self.scores_[-2]) < self.tol:
                break
    
    def _peaceman_rachford(self):
        """Peaceman-Rachford splitting."""
        x = self.x
        xhat = x
        Q = np.linalg.inv(self.AtA + np.eye(self.A.shape[1])/self.stepsize)
        for self.k in range(1, self.maxiter):
            xold = x

            x_14 = Q.dot(self.Atb + xhat/self.stepsize)
            x_12 = 2*x_14 - xhat
            x_34 = self._soft(self.stepsize*self.lamb, x_12)
            x = 2*x_34 - x_12
            xhat = self._accelerate(x, xold)
            
            self.x = x_34
            self.scores_.append(self._objective(self.x, self.x))
            if self.k > 1 and \
                    np.abs(self.scores_[-1]-self.scores_[-2]) < self.tol:
                break
    
    def _douglas_rachford(self):
        """Douglas-Rachford splitting."""
        x = self.x
        xhat = x
        Q = np.linalg.inv(self.AtA + np.eye(self.A.shape[1])/self.stepsize)
        for self.k in range(1, self.maxiter):
            xold = x

            x_14 = Q.dot(self.Atb + xhat/self.stepsize)
            x_12 = 2*x_14 - xhat
            x_34 = self._soft(self.stepsize*self.lamb, x_12)
            x = xhat + x_34 - x_14
            xhat = self._accelerate(x, xold)
            
            self.x = x_34
            self.scores_.append(self._objective(self.x, self.x))
            if self.k > 1 and \
                    np.abs(self.scores_[-1]-self.scores_[-2]) < self.tol:
                break
    
    def _admm(self):
        """ADMM."""
        x = self.x
        c = np.zeros(len(x))
        xhat = x
        Q = np.linalg.inv(self.AtA + np.eye(self.A.shape[1])/self.stepsize)
        for self.k in range(1, self.maxiter):
            xold = x

            x_12 = Q.dot(self.Atb + xhat/self.stepsize + c)
            x = self._soft(self.stepsize*self.lamb, x_12 - self.stepsize*c)
            c = c + 1./self.stepsize*(x - x_12)
            xhat = self._accelerate(x, xold)
            
            self.x = x
            self.scores_.append(self._objective(self.x, self.x))
            if self.k > 1 and \
                    np.abs(self.scores_[-1]-self.scores_[-2]) < self.tol:
                break
    
    def _admm2(self):
        """Extended version of ADMM."""
        x = self.x
        c = np.zeros(len(x))
        xhat = x
        Q = np.linalg.inv(self.AtA + np.eye(self.A.shape[1])/self.stepsize)
        for self.k in range(1, self.maxiter):
            xold = x

            x_12 = Q.dot(self.Atb + xhat/self.stepsize + c)
            x = self._soft(self.stepsize*self.lamb, x_12 - self.stepsize*c)
            c = c + 1./self.stepsize*((x+xhat)/2 - x_12)
            xhat = self._accelerate(x, xold)
            
            self.x = x
            self.scores_.append(self._objective(self.x, self.x))
            if self.k > 1 and \
                    np.abs(self.scores_[-1]-self.scores_[-2]) < self.tol:
                break

    def _cvx(self):
        """Solution with CVX."""
        x = cp.Variable(self.A.shape[1])
        lamb = cp.Parameter(nonneg=True)
        objective = cp.Minimize(0.5*cp.sum_squares(self.A*x - self.b)+\
                                lamb*cp.norm(x, 1))
        prob = cp.Problem(objective)
        lamb.value = self.lamb
        r = prob.solve(verbose=False, eps_abs=self.tol)
        self.x = x.value
        self.scores_ = [self._objective(self.x, self.x)]
        self.k = prob.solver_stats.num_iters

    def _soft(self, lamb, x):
        """Soft threshold operator."""
        return np.sign(x)*np.maximum(np.abs(x)-lamb, 0)

    def _accelerate(self, x, xold):
        """Light, heavy (or no) acceleration."""
        if self.accel == 'decaying':
            y = x + (self.k)/(self.k+self.damp)*(x - xold)
        elif self.accel == 'constant':
            y = x + (1-self.damp*np.sqrt(self.stepsize))*(x - xold)
        else:
            y = x
        return y

    def _objective(self, x, z):
        """Compute objective function."""
        f = 0.5*np.linalg.norm(self.A.dot(x) - self.b)**2
        f+= self.lamb*np.linalg.norm(z, ord=1) 
        return f

