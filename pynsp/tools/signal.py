import numpy as np
from scipy import sparse

def baseline_als(y, lamda, p, niter):
    """Asymmetric Least Squares Smoothing for Baseline fitting
    :param y: data
    :param lamda: smoothness
    :param p: assymetry
    :param niter: number of iteration
    :return:
    """
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in xrange(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lamda * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return np.asarray(z)

def baseline_fitting(data, lamda, p, niter=10):
    """Apply baseline fitting
    """
    z = baseline_als(data, lamda, p, niter)
    z = z - z[0]
    output = data - z
    return np.asarray(output)