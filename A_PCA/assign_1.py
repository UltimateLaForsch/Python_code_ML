import numpy as np


def mean_naive(X):
    """Compute the sample mean for a dataset by iterating over the dataset.

    Args:
        X: `ndarray` of shape (N, D) representing the dataset. N
        is the size of the dataset and D is the dimensionality of the dataset.
    Returns:
        mean: `ndarray` of shape (D, ), the sample mean of the dataset `X`.
    """
    # YOUR CODE HERE
    ### Uncomment and edit the code below
    #     iterate over the dataset and compute the mean vector.
    N, D = X.shape
    mean = np.zeros((D,))
    for n in range(N):
        mean = (n * mean + 1 * X[n, ]) / (n + 1)  # Update the mean vector
        pass
    return mean


def cov_naive(X):
    """Compute the sample covariance for a dataset by iterating over the dataset.

    Args:
        X: `ndarray` of shape (N, D) representing the dataset. N
        is the size of the dataset and D is the dimensionality of the dataset.
    Returns:
        ndarray: ndarray with shape (D, D), the sample covariance of the dataset `X`.
    """
    # YOUR CODE HERE
    ### Uncomment and edit the code below
    N, D = X.shape
    ### Edit the code below to compute the covariance matrix by iterating over the dataset.
    covariance = np.zeros((D, D))
    ### Update covariance
    mean_X = mean_naive(X)
    for n in range(N):
        cov_ind = np.zeros((D, D))
        for d in range(D):
            for d2 in range(D):
                cov_ind[d, d2] = (X[n, d] - mean_X[d]) * (X[n, d2] - mean_X[d2])
        covariance = (n * covariance + 1 * cov_ind) / (n + 1)
    return covariance


def mean(X):
    """Compute the sample mean for a dataset.

    Args:
        X: `ndarray` of shape (N, D) representing the dataset. N
        is the size of the dataset and D is the dimensionality of the dataset.
    Returns:
        ndarray: ndarray with shape (D,), the sample mean of the dataset `X`.
    """
    # YOUR CODE HERE
    ### Uncomment and edit the code below
    m = np.zeros((X.shape[1]))
    m = np.mean(X, axis=0)
    return m


def cov(X):
    """Compute the sample covariance for a dataset.

    Args:
        X: `ndarray` of shape (N, D) representing the dataset. N
        is the size of the dataset and D is the dimensionality of the dataset.
    Returns:
        ndarray: ndarray with shape (D, D), the sample covariance of the dataset `X`.
    """
    # YOUR CODE HERE

    # It is possible to vectorize our code for computing the covariance with matrix multiplications,
    # i.e., we do not need to explicitly
    # iterate over the entire dataset as looping in Python tends to be slow
    # We challenge you to give a vectorized implementation without using np.cov, but if you choose to use np.cov,
    # be sure to pass in bias=True.
    ### Uncomment and edit the code below


    N, D = X.shape
    ### Edit the code to compute the covariance matrix
    covariance_matrix = np.zeros((D, D))
    ### Update covariance_matrix here
    covariance_matrix = np.cov(X.T, bias=True)
    ###
    return covariance_matrix


def affine_mean(mean, A, b):
    """Compute the mean after affine transformation
    Args:
        mean: `ndarray` of shape (D,), the sample mean vector for some dataset.
        A, b: `ndarray` of shape (D, D) and (D,), affine transformation applied to x
    Returns:
        sample mean vector of shape (D,) after affine transformation.
    """
    # YOUR CODE HERE
    ### Uncomment and edit the code below


    ### Edit the code below to compute the mean vector after affine transformation
    affine_m = np.zeros(mean.shape)  # affine_m has shape (D,)
    ### Update affine_m
    affine_m = mean.dot(A.T) + b
    ###
    return affine_m


def affine_covariance(S, A, b):
    """Compute the covariance matrix after affine transformation

    Args:
        mean: `ndarray` of shape (D,), the sample covariance matrix for some dataset.
        A, b: `ndarray` of shape (D, D) and (D,), affine transformation applied to x

    Returns:
        sample covariance matrix of shape (D, D) after the transformation
    """
    # YOUR CODE HERE
    ### Uncomment and edit the code below
    ### EDIT the code below to compute the covariance matrix after affine transformation
    affine_cov = np.zeros(S.shape) # affine_cov has shape (D, D)
    affine_cov = A.dot(S).dot(A.T)
    ### Update affine_cov

    ###
    return affine_cov

# Mean and Covariance
X = np.array([[0., 1., 1.],
              [1., 2., 1.]])
print('mean - naive calc: \n', mean_naive(X))
print('mean - vectorized calc: \n', mean(X))
X2 = np.array([[0., 1.],
              [1., 2.],
              [0., 1.],
              [1., 2.]])
print('covariance - naive calc: \n', cov_naive(X2))
print('covariance - vectorized calc: \n', cov(X2))

# Affine Transformation
A = np.array([[0, 1], [2, 3]])
b = np.ones(2)
m = np.full((2,), 2)
S = np.eye(2)*2

print('A: ', A)
print('b: ', b)
print('S: ', S)
print('new given mean: ', m)
print('mean - after affine transformation calc: \n', affine_mean(m, A, b))
print('covariance - after affine transformation calc: \n', affine_covariance(S, A, b))




