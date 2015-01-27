from kmc.score_matching.gaussian_rkhs import score_matching_sym, \
    score_matching_sym_low_rank
from kmc.score_matching.kernel.incomplete_cholesky import incomplete_cholesky_gaussian
from kmc.score_matching.kernel.kernels import gaussian_kernel,\
    gaussian_kernel_grad
from kmc.scripts.trajectories.plotting import evaluate_density_grid, plot_array, \
    plot_hamiltonian_trajectory
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    D = 2
    
    # true target log density
    Sigma = np.eye(D)
    Sigma[:2, :2] = np.array([[1, .95], [.95, 1]])
    L = np.linalg.cholesky(Sigma)
    dlogq = lambda x: log_gaussian_pdf(x, Sigma=L, is_cholesky=True, compute_grad=True)
    logq = lambda x: log_gaussian_pdf(x, Sigma=L, is_cholesky=True)

    # estimate density in rkhs
    sigma = 1.
    N = 10000
    lmbda = N
    Z = L.dot(np.random.randn(D, N)).T
#     a = score_matching_sym(Z, sigma, lmbda)
    R = incomplete_cholesky_gaussian(Z, sigma, eta=0.1)["R"]
    print "Low rank dimension: %d/%d" % (R.shape[0], N)
    a = score_matching_sym_low_rank(Z, sigma, lmbda, L=R.T)
    logq = lambda x: gaussian_kernel(Z, x[np.newaxis, :])[:, 0].dot(a)
    dlogq = lambda x: a.dot(gaussian_kernel_grad(x, Z))
    
    # momentum
    logp = lambda x: log_gaussian_pdf(x)
    dlogp = lambda x: log_gaussian_pdf(x, compute_grad=True)

    # compute and plot log-density, if D==2
    if D is 2:
        Xs = np.linspace(-2, 2)
        Ys = np.linspace(-2, 2)
        G = evaluate_density_grid(Xs, Ys, logq)
    
    np.random.seed(9)
    p = np.random.randn(D)
    q = np.ones(D)
    q[:2] = np.array([-1.5, -1.55])
    num_steps=50

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plot_array(Xs, Ys, np.exp(G))
    plt.subplot(122)
    plot_hamiltonian_trajectory(q, logq, dlogq, logp, dlogp, p, num_steps=num_steps,
                                plot_H=True)
    plt.show()
