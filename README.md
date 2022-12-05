# Accelerated Proximal Splitting Methods

Code used in the paper
* [G. França, D. P. Robinson, and R. Vidal, "Gradient flows and proximal splitting methods: A unified view on accelerated and stochastic optimization," Phys. Rev. E 103, 053304 (2021)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.103.053304)

We propose several new accelerated variants of proximal splitting methods, which are powerful methods for optimization.
For instance, accelerated variants of Proximal Point, ADMM, Douglas-Rachford, Davis-Yin, and Tseng splitting are proposed.
All these methods can be seen as discretizations of the same continuous-time dynamical system, in deterministic and
stochastic settings, as summarized below:

![](https://github.com/guisf/accelerated_proximal_splitting/blob/main/figs/diagram.png)

Applying some of our proposed methods to a LASSO regression problem, we see an improved convergence
compared to the base method (these are referenced as "decaying" and "constant"):

![](https://github.com/guisf/accelerated_proximal_splitting/blob/main/figs/lasso2.png)

By solving a matrix completion problem, we can recover a matrix from few observed entries.
We illustrate this by recoverring an image where some pixels were erased:

![](https://github.com/guisf/accelerated_proximal_splitting/blob/main/figs/boats_copy.png)

We implement the accelerated algorithms with an annealing schedulle on a penalty parameter. The improved convergence rate
of some of our methods are shown below:

![](https://github.com/guisf/accelerated_proximal_splitting/blob/main/figs/boat_annealing.png)
