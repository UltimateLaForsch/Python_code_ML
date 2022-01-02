# Gauss-Seidel solution
# numeric solution for a system of linear equations
# A = coefficient matrix
# b = result vector
# x0 = x start vector
# x = solution vector

import math
import numpy as np


def gs_solution(A, b, x, e):
    error = np.dot(A, x) - b
    residual = math.sqrt(sum(error ** 2))
    n = len(b)
    loop = 1
    while residual > e:
        loop = loop + 1
        for row in range(0, n):
            if row == 0:
                bc = b[0]
                Ac = A[0, 1:n]
                xc = x[1:n]
                sub = np.inner(Ac, xc)
                x[0] = (bc - sub) / A[0, 0]
            elif row == n - 1:
                bc = b[n - 1]
                Ac = A[n - 1, 0:n - 1]
                xc = x[0:n - 1]
                sub = np.inner(Ac, xc)
                x[n - 1] = (bc - sub) / A[n - 1, n - 1]
            else:
                bc = b[row]
                Ac1 = A[row, 0:row]
                xc1 = x[0:row]
                sub1 = np.inner(Ac1, xc1)
                Ac2 = A[row, row + 1:n]
                xc2 = x[row + 1:n]
                sub2 = np.inner(Ac2, xc2)
                x[row] = (bc - sub1 - sub2) / A[row, row]
        error = np.dot(A, x) - b
        residual = math.sqrt(sum(error ** 2))
        print("\n Step: ", loop)
        print(x)
        print(residual)
    return x, loop


# Example
A = np.array([[6, 4, -1],
              [2, 5, 1],
              [-1, -1, 4]],
             dtype=np.float)
b = np.array([5, 4, -5], dtype=np.float)
x0 = np.array([0, 0, 0], dtype=np.float)
x0 = x0
e = 1e-3
x, loop = gs_solution(A, b, x0, e)
print("\n Final solution: after ", loop, " loops...")
print(x)
print("Rounded:")
rounded = lambda x: np.round(x)
x_rounded = rounded(x)
print("\n Final solution (rounded): ", x_rounded)
