import numpy as np


def projection_matrix_1d(b):
    """Compute the projection matrix onto the space spanned by `b`
    Args:
        b: ndarray of dimension (D,), the basis for the subspace

    Returns:
        P: the projection matrix
    """
    # YOUR CODE HERE
    ### Uncomment and modify the code below
    D, = b.shape
    ### Edit the code below to compute a projection matrix of shape (D,D)
    P = np.outer(b, b.T) /  b.T.dot(b)

    # You may be tempted to follow the formula and implement bb^T as b @ b.T in numpy.
    # However, notice that this b is a 1D ndarray, so b.T is an no-op. Use np.outer instead
    # to implement the outer product.
    return P


def project_1d(x, b):
    """Compute the projection matrix onto the space spanned by `b`
    Args:
        x: the vector to be projected
        b: ndarray of dimension (D,), the basis for the subspace

    Returns:
        y: ndarray of shape (D,) projection of x in space spanned by b
    """
    # YOUR CODE HERE
    ### Uncomment and modify the code below
    y = projection_matrix_1d(b).dot(x)
    return y


def projection_matrix_general(B):
    """Compute the projection matrix onto the space spanned by the columns of `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace

    Returns:
        P: the projection matrix
    """
    # YOUR CODE HERE
    ### Uncomment and modify the code below


    P = B.dot(np.linalg.inv(B.T.dot(B))).dot(B.T)
    return P

def project_general(x, B):
    """Compute the projection matrix onto the space spanned by the columns of `B`
    Args:
        x: ndarray of dimension (D, 1), the vector to be projected
        B: ndarray of dimension (D, M), the basis for the subspace

    Returns:
        p: projection of x onto the subspac spanned by the columns of B; size (D, 1)
    """
    # YOUR CODE HERE
    # Uncomment and modify the code below
    p = projection_matrix_general(B).dot(x)
    return p


pm_1d = projection_matrix_1d(np.array([1, 2, 2]))
print(pm_1d)
p_1d = project_1d(np.ones(3), np.array([1, 2, 2]))
print(p_1d)

