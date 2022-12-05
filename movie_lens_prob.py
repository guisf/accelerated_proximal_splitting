
import numpy as np
import pickle
import matplotlib.pyplot as plt

from matrix_completion import MatrixCompletion

"""
M = np.zeros((6040, 3952))

for row in open('ml-1m/ratings.dat'):
    user_id, movie_id, rating, timestamp = row.strip().split('::')
    M[int(user_id)-1, int(movie_id)-1] = float(rating)

pickle.dump(M, open('movie_lens_mat.pickle', 'wb'))
"""

M = pickle.load(open('movie_lens_mat.pickle', 'rb'))


Mask = np.ones(M.shape)
Mask[np.where(M == 0)] = 0

mc = MatrixCompletion(mu=35, lower=0.0, higher=5.0,
                      method='dy', stepsize=1, maxiter=10, tol=1e-3)
mc.fit(M, Mask)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.show()
