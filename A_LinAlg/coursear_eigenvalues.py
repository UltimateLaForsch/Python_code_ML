import numpy as np
import math
import pandas as pd
# Eigenvalues
M = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 3]])
M2 = np.array([[4, -5, 6],
              [7, -8, 6],
              [3/2, -1/2, -2]])
M3 = np.array([[0,0,0,1],
              [1,0,0,0],
              [0,1,0,0],
               [0,0,1,0]])
M4 = np.array([[0.1,0.1,0.1,0.7],
              [0.7,0.1,0.1,0.1],
              [0.1,0.7,0.1,0.1],
               [0.1,0.1,0.7,0.1]])
M5= np.array([[0, 1],
              [1, 0]])
vals, vecs = np.linalg.eig(M5)
vals2 = pd.DataFrame(vals)
rounded = lambda x: round(x)
vals_rounded = vals2.apply(rounded)
print("Eigenvalues:")
print(vals_rounded)
print("Eigenvectors:")
vecs2 = pd.DataFrame(vecs)
vecs_rounded = vecs2.apply(rounded)
print(vecs_rounded)


