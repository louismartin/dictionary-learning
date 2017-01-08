import numpy as np
from utils import scale


# ----- K-SVD -----
def dictionary_update_ksvd(Y, D, X):
    (signal_size, n_atoms) = D.shape
    (_, n_samples) = Y.shape

    for k in range(n_atoms):
        dk = D[:, k].reshape((signal_size, 1))
        xk = X[k, :].reshape((1, n_samples))  # Careful, this is the kth ROW

        # Error part that does not depend on dk
        Ek = Y - (np.dot(D, X) - (np.dot(dk, xk)))

        # Samples that use atom dk
        omega_k = np.where(xk != 0)[1]
        xk_R = xk[:, omega_k]
        Ek_R = Ek[:, omega_k]
        U, s, V = np.linalg.svd(Ek_R)

        # Update dk column and xk row
        D[:, k] = U[:, 0]
        X[k, omega_k] = s[0] * V[:, 0]

    return D, X


# ----- Online matrix factorization -----
def sparse_code_lasso(Y, D, model):
    '''
    Sparse code data Y using dictionary D using lasso linear regression
    Model is a sklearn.linear_model.Lasso lasso model
    '''
    X = model.fit(D, Y).coef_.T
    return X


def dictionary_update_omf(D, A, B):
    '''
    Algorithm from "Online Learning for Matrix Factorization and SparseCoding
    Update the dictionary column by column.
    Denoting k the number of atoms in the dictionary and m the size of the
    signal, we have:

    Args:
        D: dictionary of size (m,k)
        A: Matrix of size (k,k)
        B: Matrix of size (m,k)
    Returns:
        D: Updated dictionary of size (m,k)
    '''
    (m, k) = D.shape
    assert A.shape == (k, k)
    assert B.shape == (m, k)

    for j in range(k):
        uj = (B[:, j] - np.dot(D, A[:, j])) + D[:, j]
        if A[j, j] != 0:
            uj /= A[j, j]
        else:
            # TODO: What to do when A[j,j] is 0 ?
            pass
        D[:, j] = 1/max(np.linalg.norm(uj), 1)*uj
    return D


# ----- Forward-Backward -----
def ProjX(X, k):
    ''' Sparsity projection, keeps the k largest coefficients '''
    X = X * (abs(X) >= np.sort(abs(X), axis=0)[-k, :])
    return X


def ProjC(D):
    ''' Dictionary projection, scales the atoms '''
    D = scale(D)
    return D


def sparse_code_fb(Y, D, X, sparsity=4, n_iter=100):
    '''
    Sparse code data Y using dictionary D using a forward backward iterative
    scheme (projected block coordinate gradient descent).
    '''
    tau = 1/np.linalg.norm(np.dot(D, D.T))
    for i in range(n_iter):
        R = np.dot(D, X) - Y
        X = ProjX(X - tau * np.dot(D.T, R), sparsity)
    return X


def dictionary_update_fb(Y, D, X, n_iter=50):
    tau = 1/np.linalg.norm(np.dot(X, X.T))
    for i in range(n_iter):
        R = np.dot(D, X) - Y
        D = ProjC(D - tau * np.dot(R, X.T))
    return D
