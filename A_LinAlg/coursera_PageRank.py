import numpy as np
import numpy.linalg as la
# from readonly.PageRankFunctions import *
np.set_printoptions(suppress=True)

# Replace the ??? here with the probability of clicking a link to each website when leaving Website F (FaceSpace).
L = np.array([[0,   1/2, 1/3, 0, 0,   0],
              [1/3, 0,   0,   0, 1/2, 0],
              [1/3, 1/2, 0,   1, 0,   0.5],
              [1/3, 0,   1/3, 0, 1/2, 0.5],
              [0,   0,   0,   0, 0, 0],
              [0,   0,   1/3, 0, 0, 0]])

# Gets the eigenvalues and vectors
eVals, eVecs = la.eig(L)
# Orders them by their eigenvalues
order = np.absolute(eVals).argsort()[::-1]
eVals = eVals[order]
eVecs = eVecs[:,order]
# Sets r to be the principal eigenvector
r = eVecs[:, 0]
# Make this eigenvector sum to one, then multiply by 100 Procrastinating Pats
solution1 = 100 * np.real(r / np.sum(r))
print(solution1)

# Let's now try to get the same result using the Power-Iteration method
solution2 = 100 * np.ones(6) / 6 # Sets up this vector (6 entries of 1/6 × 100 each)
for i in np.arange(100) : # Repeat 100 times
    solution2 = L @ solution2
print(solution2)

# even better, we can keep running until we get to the required tolerance.
solution3 = 100 * np.ones(6) / 6 # Sets up this vector (6 entries of 1/6 × 100 each)
last_solution = solution3
solution3 = L @ solution3
i = 0
while la.norm(last_solution - solution3) > 0.01 :
    last_solution = solution3
    solution3 = L @ solution3
    i += 1
print(str(i) + " iterations to convergence.")
print(solution3)

 # We'll call this one L2, to distinguish it from the previous L.
L2 = np.array([[0,   1/2, 1/3, 0, 0, 0, 0],
               [1/3, 0,   0,   0, 1/2, 0, 0],
               [1/3, 1/2, 0,   1, 0, 1/3, 0],
               [1/3, 0,   1/3, 0, 1/2, 1/3, 0],
               [0,   0,   0,   0, 0, 0, 0],
               [0,   0,   1/3, 0, 0, 0, 0],
               [0,   0,   0,   0, 0, 1/3, 1]])

r = 100 * np.ones(7) / 7 # Sets up this vector (6 entries of 1/6 × 100 each)
lastR = r
r = L2 @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = L2 @ r
    i += 1
print(str(i) + " iterations to convergence.")
print(r)

# Damping parameter
# Feel free to play with this parameter after running the code once.
d = 0.5
M = d * L2 + (1-d)/7 * np.ones([7, 7]) # np.ones() is the J matrix, with ones for each entry.

r = 100 * np.ones(7) / 7 # Sets up this vector (6 entries of 1/6 × 100 each)
lastR = r
r = M @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = M @ r
    i += 1
print(str(i) + " iterations to convergence.")
print(r)
