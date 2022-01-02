import numpy as np
np.set_printoptions(suppress=True)

A = np.array([[1, 0, 0.33333], [0, 1, -0.25]])
r_row = np.array([[6, 2, 3]])
r_col = r_row.T
r = np.dot(A, r_col)
print(r)

# now apply to list of vectors (~matrix)
r_mat = np.array([[5, -1, -3, 7], [4, -4, 1, -2], [9, 3, 0, 12]])
R = np.dot(A, r_mat).round(2)
print(R)

