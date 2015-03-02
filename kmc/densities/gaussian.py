import numpy as np
import scipy as sp

def log_gaussian_pdf(x, mu=None, Sigma=None, is_cholesky=False, compute_grad=False):
    if mu is None:
        mu = np.zeros(len(x))
    if Sigma is None:
        Sigma = np.eye(len(mu))
    
    if is_cholesky is False:
        L = np.linalg.cholesky(Sigma)
    else:
        L = Sigma
    
    assert len(x) == Sigma.shape[0]
    assert len(x) == Sigma.shape[1]
    assert len(x) == len(mu)
    
    # solve y=K^(-1)x = L^(-T)L^(-1)x
    x = np.array(x - mu)
    y = sp.linalg.solve_triangular(L, x.T, lower=True)
    y = sp.linalg.solve_triangular(L.T, y, lower=False)
    
    if not compute_grad:
        log_determinant_part = -np.sum(np.log(np.diag(L)))
        quadratic_part = -0.5 * x.dot(y)
        const_part = -0.5 * len(L) * np.log(2 * np.pi)
        
        return const_part + log_determinant_part + quadratic_part
    else:
        return -y

def sample_gaussian(N, mu=np.zeros(2), Sigma=np.eye(2), is_cholesky=False):
    D = len(mu)
    assert len(mu.shape) == 1
    assert len(Sigma.shape) == 2
    assert D == Sigma.shape[0]
    assert D == Sigma.shape[1]
    
    if is_cholesky is False:
        L = np.linalg.cholesky(Sigma)
    else:
        L = Sigma
    
    return L.dot(np.random.randn(D, N)).T + mu
    