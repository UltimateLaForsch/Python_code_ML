import numpy as np
import scipy
import scipy.stats


def normalize(X):
    """Normalize the given dataset X to have zero mean.
    Args:
        X: ndarray, dataset of shape (N,D)

    Returns:
        (Xbar, mean): tuple of ndarray, Xbar is the normalized dataset
        with mean 0; mean is the sample mean of the dataset.
    """
    # YOUR CODE HERE
    ### Uncomment and modify the code below
    N, D = X.shape
    mu = np.mean(X, axis=0)  # <-- EDIT THIS, compute the mean of X
    Xbar = X - mu            # <-- EDIT THIS, compute the normalized data Xbar
    return Xbar, mu


# GRADED FUNCTION: DO NOT EDIT THIS LINE
def eig(S):
    """Compute the eigenvalues and corresponding eigenvectors
        for the covariance matrix S.
    Args:
        S: ndarray, covariance matrix

    Returns:
        (eigvals, eigvecs): ndarray, the eigenvalues and eigenvectors

    Note:
        the eigenvals and eigenvecs should be sorted in descending
        order of the eigen values
    """
    # YOUR CODE HERE
    # Uncomment and modify the code below
    # Compute the eigenvalues and eigenvectors
    # You can use library routines in `np.linalg.*`
    # https://numpy.org/doc/stable/reference/routines.linalg.html
    # for this
    eigvals, eigvecs = np.linalg.eigh(S)
    # eigvals = np.linalg.eigvals(S)
    # eigvecs = np.linalg.eig(S)
    # The eigenvalues and eigenvectors need to be
    # sorted in descending order according to the eigenvalues
    # We will use `np.argsort` (https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html)
    # to find a permutation of the indices
    # of eigvals that will sort eigvals in ascending order and
    # then find the descending order via [::-1], which reverse
    # the indices
    sort_indices = np.argsort(eigvals)[::-1]
    # Notice that we are sorting the columns (not rows) of
    # eigvecs since the columns represent the eigenvectors.
    return eigvals[sort_indices], eigvecs[:, sort_indices]


def projection_matrix(B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace

    Returns:
        P: the projection matrix
    """
    # YOUR CODE HERE
    # Uncomment and modify the code below
    P = B@(np.linalg.inv(B.T@(B)))@(B.T)
    return P


def PCA(X, num_components):
    """
    Args:
        X: ndarray of size (N, D), where D is the dimension of the data,
           and N is the number of datapoints
        num_components: the number of principal components to use.
    Returns:
        the reconstructed data, the sample mean of the X, principal values
        and principal components
    """

    # YOUR CODE HERE
    # your solution should take advantage of the functions you have implemented above.
    ### Uncomment and modify the code below
    # first perform normalization on the digits so that they have zero mean and unit variance
    X_normalized, mean = normalize(X)
    # Then compute the data covariance matrix S
    N, D = X.shape
    # S = X_normalized.T.dot(X_normalized) / N
    S = np.cov(X_normalized, rowvar=False, bias=True)
    # Next find eigenvalues and corresponding eigenvectors for S
    eig_vals, eig_vecs = eig(S)
    # Take the top `num_components` of eig_vals and eig_vecs,
    # This will be the corresponding principal values and components
    # principal_vals, principal_components = np.take(eig_vals, num_components), np.take(eig_vecs, num_components)
    principal_values, principal_components = eig_vals[:num_components], eig_vecs[:, :num_components]
    # reconstruct the data from the using the basis spanned by the principal components
    # Notice that we have subtracted the mean from X so make sure that you add it back
    # to the reconstructed data
    reconst_1 = projection_matrix(principal_components).dot(X_normalized.T)
    reconst = reconst_1.T + mean
    return reconst, mean, principal_values, principal_components


def PCA_high_dim(X, num_components):
    """Compute PCA for small sample size but high-dimensional features.
    Args:
        X: ndarray of size (N, D), where D is the dimension of the sample,
           and N is the number of samples
        num_components: the number of principal components to use.
    Returns:
        X_reconstruct: (N, D) ndarray. the reconstruction
        of X from the first `num_components` pricipal components.
    """
    # YOUR CODE HERE
    # Uncomment and modify the code below
    N, D = X.shape
    # Normalize the dataset
    X_normalized, mean = normalize(X)
    # Find the covariance matrix
    M = np.dot(X_normalized, X_normalized.T) / N
    # Next find eigenvalues and corresponding eigenvectors for M
    eig_vals_M, eig_vecs_M = eig(M)
    # Compute the eigenvalues and eigenvectors for the original system
    eig_vals_S = eig_vals_M
    c = eig_vecs_M  # using variables as in the lecture
    eig_vecs_S = np.cov(X_normalized.T @ c, bias=True)
    S = X_normalized.T.dot(X_normalized) / N
    eig_vecs_S = S.dot(X_normalized.T).dot(c)
    # Normalize the eigenvectors to have unit-length
    # eig_vecs_S = _normalize_eigenvectors(eig_vecs_S_notNorm)
    eig_vecs_S_norm = eig_vecs_S / np.linalg.norm(eig_vecs_S)
    # Take the top `num_components` of the eigenvalues / eigenvectors
    # as the principal values and principal components
    principal_values = eig_vals_S[:num_components]
    principal_components = eig_vecs_S_norm[:, :num_components]
    # reconstruct the images from the lower dimensional representation
    # Remember to add back the sample mean
    reconst_1 = projection_matrix(principal_components).dot(X_normalized.T)
    reconst = reconst_1.T + mean
    return reconst, mean, principal_values, principal_components


def _normalize_eigenvectors(B):
    # Normalize eigenvectors to have unit length
    # Also flip the direction of the eigenvector based on
    # the first element
    B_normalized = B / np.linalg.norm(B, axis=0)
    for i in range(B.shape[1]):
        if (B_normalized[0, i] < 0):
            B_normalized[:, i] = -B_normalized[:, i]
    return B_normalized


X0 = np.array([[0, 0.0],
               [1.0, 1.0],
               [2.0, 2.0]])
X0_normalize, X0_mean = normalize(X0)
print("mean: \n", X0_mean)
print("normalized dataset \n", X0_normalize)

A = np.array([[3, 2], [2, 3]])
actual_eigenvalues, actual_eigenvectors = eig(A)
print("Eigenvalues: \n", actual_eigenvalues)
print("Eigenvectors: \n", actual_eigenvectors)

B = np.array([[1, 0],
              [1, 1],
              [1, 2]])
proj_matrix =  projection_matrix(B)
print("Projection Matrix: \n", proj_matrix)

D = 2
N = 10
# Generate a dataset X from a 2D Gaussian distribution
mvn = scipy.stats.multivariate_normal(
    mean=np.ones(D, dtype=np.float64),
    cov=np.array([[1, 0.8], [0.8, 1]], dtype=np.float64)
)

X = mvn.rvs((N,), random_state=np.random.RandomState(0))
reconst, m, pv, pc = PCA(X, 1)
print("PCA: \n", reconst, m, pv, pc)
print("pv: \n" , pv)
print(pv.shape)

random = np.random.RandomState(0)
# Generate some random data
X = random.randn(5, 4)
pca_rec, pca_mean, pca_pvs, pca_pcs = PCA(X, 2)
pca_hd_rec, pca_hd_mean, pca_hd_pvs, pca_hd_pcs = PCA_high_dim(X, 2)
print("PCA: \n", pca_rec, pca_mean, pca_pvs, pca_pcs)
print("PCA high dim: \n", pca_hd_rec, pca_hd_mean, pca_hd_pvs, pca_hd_pcs)
