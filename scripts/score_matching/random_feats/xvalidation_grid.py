from itertools import product
from multiprocessing.pool import Pool

from kmc.densities.gaussian import sample_gaussian
from kmc.score_matching.random_feats.gaussian_rkhs import xvalidate,\
    sample_basis
import matplotlib.pyplot as plt
import numpy as np


def fun(sigma_lmbda, m=100, num_repetitions=3):
    log2_sigma = sigma_lmbda[0]
    log2_lmbda = sigma_lmbda[1]
    
    sigma = 2**log2_sigma
    gamma = 0.5*(sigma**2)
    lmbda = 2**log2_lmbda
    
    omega, u = sample_basis(D, m, gamma)
    
    folds = [xvalidate(Z, lmbda, omega, u) for _ in range(num_repetitions)]
    J = np.mean(folds)
    J_std = np.std(folds)
    print("fun: log2_sigma=%.2f, log_lmbda=%.2f, J(a)=%.2f" % (log2_sigma, log2_lmbda, J))
    return J, J_std

if __name__ == "__main__":
    D = 2
    
    # true target log density
    Sigma = np.diag(np.linspace(0.01, 1, D))
    Sigma[:2, :2] = np.array([[1, .95], [.95, 1]])
    Sigma = np.eye(D)
    L = np.linalg.cholesky(Sigma)

    # estimate density in rkhs
    N = 200
    mu = np.zeros(D)
    np.random.seed(0)
    Z = sample_gaussian(N, mu, Sigma=L, is_cholesky=True)
#     print np.sum(Z) * np.std(Z) * np.sum(Z**2) * np.std(Z**2)
    
    resolution = 10
    log2_sigmas = np.linspace(-3, 3, resolution)
    log2_lambdas = np.linspace(-15, 3, resolution)
     
    Js = np.zeros((len(log2_sigmas), len(log2_lambdas)))
    Js_std = np.zeros(Js.shape)
    pool = Pool()
    chunksize = 1
    for ind, res in enumerate(pool.imap(fun, product(log2_sigmas, log2_lambdas)), chunksize):
        Js.flat[ind - 1], Js_std.flat[ind - 1] = res
    
    im = plt.pcolor(log2_lambdas, log2_sigmas, np.log2(Js-Js.min() + 1))
    plt.title(r"log of X-validated $J(\alpha)$")
    plt.xlabel(r"$\log_2 \lambda$")
    plt.ylabel(r"$\log_2 \sigma$")
     
    min_idx = np.unravel_index(Js.argmin(), Js.shape)
    log2_sigma, log2_lmbda = log2_sigmas[min_idx[0]], log2_lambdas[min_idx[1]]
    plt.plot([log2_lmbda], [log2_sigma], "m*", markersize=20)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
