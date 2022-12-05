"""Nonnegative Matrix Completion with several proximal based algorithms
and their accelerated variants.
   
We solve the problem
$$
\min \mu \| X \|_* + \mathbb{I}_{a,b}(X) 
        + 1/2\| \mathcal{P}_\Omega( X - M) \|^2
$$

"""

import numpy as np
from scipy.sparse.linalg import svds


class MatrixCompletion:
    """Nonnegative matrix completion with (accelerated) proximal methods.
    One can choose different optimizers to solve the problem.
    
    Features
    --------
    M_ : array
        final solution with completed matrix
    score_ : float
        final value of the objective function
    rank_ : int
        estimated rank
    Xs : list of np.array
        list containing the estimated matrix, one per iteration
    scores : list 
        all values of the objective function per iteration
    ranks : list
        ranks history

    """

    def __init__(self, mu=1, lower=0, higher=100, 
                 M=None, Mask=None, Mobs=None, rank_est=None,
                 method='dy', stepsize=1, accel='', damp=3, 
                 maxiter=1000, tol=1e-4):
        """
        Parameters
        ----------
        mu : [float, None]
            regularization constant; if None will be chosen automaticaly
        lower : float
            lower range for "nonnegative" entries
        higher : float
            righer range for "nonnegative" entries
        M : 2D array (optional)
            ground truth
        Mask : 2D array
            support of observed entries
        Mobs : 2D array
            observed matrix
        rank_est : int
            estimate of rank to compute truncated SVD (None for full SVD)
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
        """
        self.mu = mu # parameter of nuclear norm
        self.lower = lower # lower box constraint
        self.higher = higher # higher box constraint
        self.method = method # optimization method
        self.rank_est = rank_est # rank for truncated SVD
        self.stepsize = stepsize # optimizer stepsize
        self.accel = accel # type of acceleration {decaying, constant, None}
        self.damp = damp # damping coefficient constant
        self.maxiter = maxiter # maximum number iterations
        self.tol = tol # tolerance error
        
        self._reset() # define output variables
        self._set_data(M, Mask, Mobs) # potentially define input data
    
    def _reset(self):
        self.k = 0 # number of iterations (keep adding, works with annealing)
        self.scores = [] # objective history
        self.score_ = np.inf # final objective value
        self.Xs = [] # solution history 
        self.X = None # final solution (used for initialization in annealing)
        self.ranks = [] # ranks history
        self.rank_ = 0 # final rank
        self.M_ = None  # final solution estimate

    def _set_data(self, M=None, Mask=None, Mobs=None):
        # we do not use M (ground truth) anywhere, however it can set Mobs
        # if we do pass it. It uses M however to compute total error
        self.M = M
        self.Mask = Mask
        self.Mobs = Mobs
        if type(self.M) is np.ndarray and type(self.Mask) is np.ndarray:
            self.Mobs = self.Mask*self.M

    def _test_data_is_set(self):
        # check if there is data to solve the problem
        if self.Mobs is None or self.Mask is None:
            raise ValueError("No data is set")

    def fit(self, Mobs, Mask, M=None):
        """
        Call an optimizer and solve the optimization problem.

        Parameters
        ----------
        Mobs : 2D array
            matrix with observed entries
        Mask : 2D array
            binary matrix containing the support of observations
        M : 2D array (optional)
            ground truth
        
        """
        self._reset()
        self._set_data(M=M, Mask=Mask, Mobs=Mobs)
        self._test_data_is_set()
        self.X = np.zeros(self.Mobs.shape)
        self.scores.append(self._objective(self.X))
        self._call_solver()
        self._set_final_solution()
       
    def _set_final_solution(self):
        # set some variables for the final solution
        UnobMask = np.ones(self.Mobs.shape, dtype=int)-self.Mask
        self.M_ = self.Mobs + UnobMask*self.X
        self.score_ = self._objective(self.M_)
        self.rank_ = self.ranks[-1]
    
    def _call_solver(self):
        # solve the problem
        if self.method == 'dy':
            self._davis_yin()
        elif self.method == 'admm':
            self._admm()
        elif self.method == 'admm2':
            self._admm2()
        else:
            raise ValueError('No optimization method specified.')

    def solve(self, method='dy', stepsize=1, accel='', damp=3, 
                maxiter=1000, tol=1e-4):
        """Call optimization solver. Data must be previously set.
        Same variables as __init___.
        
        """
        self.method = method
        self.stepsize = stepsize
        self.accel = accel
        self.damp = damp
        self.maxiter = maxiter
        self.tol = tol
        self.fit(self.Mobs, self.Mask, self.M)

    def solve_annealing(self, mus, method='dy', stepsize=1, accel='', damp=3,
                            maxiter=1000, tol=1e-4):
        """Solve the problem with an annealing schedulle.
        Assumes data is set.
        
        Parameters
        ----------
        mus : list
            list of parameter mu (nuclear norm coefficient)

        for the other parameters see __init__
        
        """
        self.method = method
        self.stepsize = stepsize
        self.accel = accel
        self.damp = damp
        self.maxiter = maxiter
        self.tol = tol
        self._reset()
        
        self._test_data_is_set()
        self.X = np.zeros(self.Mobs.shape)
        self.scores.append(self._objective(self.X))
        for mu in mus:
            self.mu = mu
            self._call_solver()
        self._set_final_solution()

    def _davis_yin(self):
        """Davis-Yin solver."""
        X = self.X
        Xhat = X
        for k in range(1, self.maxiter):
            Xold = X
            
            X14, rank = self._SVDsoft(self.mu*self.stepsize, Xhat)
            X12 = 2*X14 - Xhat
            X34 = self._clip(X12 - self.stepsize*(self.Mask*X14-self.Mobs))
            X = Xhat + X34 - X14
            Xhat = self._accelerate(X, Xold)
            
            self.Xs.append(X)
            self.ranks.append(rank)
            self.scores.append(self._objective(X))
            self.X = X

            error = np.linalg.norm(X-Xold)/np.max([1, np.linalg.norm(Xold)])
            if error <= self.tol:
                break
            self.k += 1
    
    def _admm(self):
        """ADMM solver."""
        X = self.X
        Xhat = X
        C = np.zeros(X.shape)
        for k in range(1, self.maxiter):
            Xold = X
            
            X12, rank = self._SVDsoft(self.mu*self.stepsize, 
                Xhat-self.stepsize*(self.Mask*Xhat-self.Mobs)+self.stepsize*C)
            X = self._clip(X12 - self.stepsize*C)
            C = C + 1/self.stepsize*(X - X12)
            Xhat = self._accelerate(X, Xold)
            
            self.Xs.append(X)
            self.ranks.append(rank)
            self.scores.append(self._objective(X))
            self.X = X
            
            error = np.linalg.norm(X-Xold)/np.max([1, np.linalg.norm(Xold)])
            if error <= self.tol:
                break
            self.k += 1
                
    def _admm2(self):
        """Modified ADMM solver."""
        X = self.X
        Xhat = X
        C = np.zeros(X.shape)
        for k in range(1, self.maxiter):
            Xold = X
            
            X12, rank = self._SVDsoft(self.mu*self.stepsize, 
                Xhat-self.stepsize*(self.Mask*Xhat-self.Mobs)+self.stepsize*C)
            X = self._clip(X12 - self.stepsize*C)
            C = C + 1/self.stepsize*((X+Xhat)/2 - X12)
            Xhat = self._accelerate(X, Xold)
            
            self.Xs.append(X)
            self.ranks.append(rank)
            self.scores.append(self._objective(X))
            self.X = X

            error = np.linalg.norm(X-Xold)/np.max([1, np.linalg.norm(Xold)])
            if error <= self.tol:
                break
            self.k += 1

    def _SVDsoft(self, tau, X):
        """SVD soft threshold operator."""
        if self.rank_est == None:
            U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
        else:
            U, sigma, Vt = svds(X, k=self.rank_est)
        sigma = np.maximum(sigma-tau, 0)
        rank = len(sigma[sigma>0])
        return U.dot(np.diag(sigma).dot(Vt)), rank

    def _clip(self, x):
        """Projections into a box."""
        return np.maximum(self.lower, np.minimum(x, self.higher))

    def _accelerate(self, x, xold):
        """Decaying, constant (or no) acceleration."""
        if self.accel == 'decaying':
            y = x + (self.k)/(self.k+self.damp)*(x - xold)
        elif self.accel == 'constant':
            y = x + (1-self.damp*np.sqrt(self.stepsize))*(x - xold)
        else:
            y = x
        return y

    def _objective(self, X):
        """Compute objective function."""
        f = self.mu*np.linalg.norm(X, ord='nuc')
        f+= 0.5*np.linalg.norm(self.Mask*(X-self.Mobs)) 
        return f

    def total_error_history(self):
        """Assumes the matrix self.M contains the true matrix."""
        normM = np.linalg.norm(self.M)
        return [np.linalg.norm(self.M - X)/normM for X in self.Xs]

    def total_error(self):
        """Assumes the matrix self.M contains the true matrix."""
        return np.linalg.norm(self.M - self.M_)/np.linalg.norm(self.M)
    
def make_support(shape, m):
    """Creates support for a matrix with uniform distribution.

    Parameters 
    ----------
    shape: (rows, columns)
        m: number of nonzero entries
    Output
    ------
    Binary array of dimension 'shape' defining the support of observations

    """
    total = shape[0]*shape[1]
    omega = np.zeros(total, dtype=int)
    ids = np.random.choice(range(total), m, replace=False)
    omega[ids] = 1
    omega = omega.reshape(shape)
    return omega

def generate_data(n, r, sampling_ratio):
    """Generate data for matrix completion.
    Parameters
    ----------
    n : int
        dimension of the matrix
    r : int
        true rank of the matrix (<= n)
    sampling_ratio: float
        ratio for the number of observed entries
    """
    A = np.random.normal(loc=3, scale=1, size=(n,r))
    B = np.random.normal(loc=3, scale=1, size=(n,r))
    M = A.dot(B.T)
    number_nonzero = int(sampling_ratio*np.prod(M.shape))
    hardness = r*(np.sum(M.shape)-r)/number_nonzero
    Mask = make_support(M.shape, number_nonzero)
    Mobs = Mask*M
    return M, Mobs, Mask, hardness

def mu_schedulle(Mobs, eta=0.25, mubar=1e-8):
    """Schedulle for the parameter of the nuclear norm."""
    mu = eta*np.linalg.norm(Mobs)
    mus = []
    while mu > mubar:
        mu = np.max([eta*mu, mubar])
        mus.append(mu)
    return mus

    
###############################################################################
if __name__ == "__main__":

    from prettytable import PrettyTable
    import matplotlib.pyplot as plt

    M, Mobs, Mask, hardness = generate_data(100, 5, 0.4)
    sigma = Mobs.std()
    a = Mobs[np.where(Mobs!=0)].min()-0.5*sigma
    b = Mobs.max()+0.5*sigma
    print('hardness=%.2f'%(hardness))
    print('True range: %s'%([M[np.where(M!=0)].min(), M.max()]))
    print('Estimated range: %s'%([a, b]))

    t = PrettyTable()
    t.field_names = ['algorithm', 'error', 'rank', 'iter']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')

    mc = MatrixCompletion(mu=3.5, lower=a, higher=b, M=M, Mobs=Mobs, Mask=Mask)
    
    mc.solve(method='admm2', damp=1, maxiter=5000, tol=1e-8)
    t.add_row(['DY', mc.total_error(), mc.rank_, mc.k])
    ax.plot(mc.total_error_history(), label='dy')
    
    mc.solve(method='admm2', accel='decaying', damp=3, maxiter=5000, tol=1e-8)
    t.add_row(['DY (decaying)', mc.total_error(), mc.rank_, mc.k])
    ax.plot(mc.total_error_history(), label='dy light')
    
    mc.solve(method='admm2', accel='constant', damp=0.1, maxiter=5000, 
                tol=1e-8)
    t.add_row(['DY (constant)', mc.total_error(), mc.rank_, mc.k])
    ax.plot(mc.total_error_history(), label='dy heavy')
   
    """
    mus = mu_schedulle(Mobs, eta=0.25, mubar=1e-8)

    mc.solve_annealing(mus, method='dy', maxiter=5000, tol=1e-10)
    t.add_row(['DY (annealing)', mc.total_error(), mc.rank_, mc.k])
    ax.plot(mc.total_error_history(), label='dy anneal')
    
    mc.solve_annealing(mus, method='dy', accel='decaying', damp=3, 
                        maxiter=5000,tol=1e-10)
    t.add_row(['DY (annealing, decaying)', mc.total_error(), mc.rank_, mc.k])
    ax.plot(mc.total_error_history(), label='dy decaying anneal')
    
    mc.solve_annealing(mus, method='dy', accel='constant', damp=0.5, 
                        maxiter=5000,tol=1e-10)
    t.add_row(['DY (annealing, constant)', mc.total_error(), mc.rank_, mc.k])
    ax.plot(mc.total_error_history(), label='dy constant anneal')
    """

    ax.legend(loc=0)

    print(t)
    fig.savefig('mat_test.pdf')


