import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
# from readonly.PageRankFunctions import *
np.set_printoptions(suppress=True)


def generate_internet(n) :
    c = np.full([n,n], np.arange(n))
    c = (abs(np.random.standard_cauchy([n,n])/2) > (np.abs(c - c.T) + 1)) + 0
    c = (c+1e-10) / np.sum((c+1e-10), axis=0)
    return c


# GRADED FUNCTION
# Complete this function to provide the PageRank for an arbitrarily sized internet.
# I.e. the principal eigenvector of the damped system, using the power iteration method.
# (Normalisation doesn't matter here)
# The functions inputs are the linkMatrix, and d the damping parameter - as defined in this worksheet.
# (The damping parameter, d, will be set by the function - no need to set this yourself.)
def pageRank(linkMatrix, d):
    n = linkMatrix.shape[0]
    M = d * linkMatrix + (1 - d) / n * np.ones([n, n])
    r = 100 * np.ones(n) / n  
    lastR = r
    r = M @ r
    i = 0
    while la.norm(lastR - r) > 0.01:
        lastR = r
        r = M @ r
        i += 1
    print(str(i) + " iterations to convergence.")
    return r


# Use the following function to generate internets of different sizes.
L = generate_internet(10)
r = pageRank(L, 1)
print(r)

# You may wish to view the PageRank graphically.
# This code will draw a bar chart, for each (numbered) website on the generated internet,
# The height of each bar will be the score in the PageRank.
# Run this code to see the PageRank for each internet you generate.
# Hopefully you should see what you might expect
# - there are a few clusters of important websites, but most on the internet are rubbish!
r = pageRank(generate_internet(100), 0.9)
plt.bar(np.arange(r.shape[0]), r);
plt.show()