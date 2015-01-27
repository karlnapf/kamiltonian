from itertools import product
from multiprocessing.pool import Pool

from kmc.densities.gaussian import sample_gaussian
from kmc.score_matching.gaussian_rkhs import xvalidate
from kmc.score_matching.kernel.kernels import gaussian_kernel
import matplotlib.pyplot as plt
import numpy as np


def fun(sigma_lmbda, num_repetitions=1):
    log2_sigma = sigma_lmbda[0]
    log2_lmbda = sigma_lmbda[1]
    
    sigma = 2**log2_sigma
    lmbda = 2**log2_lmbda
    K = gaussian_kernel(Z, sigma=sigma)
    folds = [xvalidate(Z, num_folds, sigma, lmbda, K) for _ in range(num_repetitions)]
    J = np.mean(folds)
    J_std = np.std(folds)
    print "fun: log2_sigma=%.2f, log_lmbda=%.2f, J(a)=%.2f" % (log2_sigma, log2_lmbda, J)
    return J, J_std

if __name__ == "__main__":
    D = 2
    
    # true target log density
    Sigma = np.diag(np.linspace(0.01, 1, D))
    Sigma[:2, :2] = np.array([[1, .95], [.95, 1]])
    Sigma = np.eye(D)
    L = np.linalg.cholesky(Sigma)

    # estimate density in rkhs
    N = 500
    mu = np.zeros(D)
    Z = sample_gaussian(N, mu, Sigma=L, is_cholesky=True)
    
    num_folds = 5
    resolution = 10
    log2_sigmas = np.linspace(-10, 10, resolution)
    log2_lambdas = np.linspace(-10, 10, resolution)
     
    Js = np.zeros((len(log2_sigmas), len(log2_lambdas)))
    Js_std = np.zeros(Js.shape)
    pool = Pool(2)
    chunksize = 1
    for ind, res in enumerate(pool.imap(fun, product(log2_sigmas, log2_lambdas)), chunksize):
        Js.flat[ind - 1], Js_std.flat[ind - 1] = res
     
    im = plt.pcolor(log2_lambdas, log2_sigmas, Js)
    plt.title(r"X-validated $J(\alpha)$")
    plt.xlabel(r"$\log_2 \lambda$")
    plt.ylabel(r"$\log_2 \sigma$")
     
    min_idx = np.unravel_index(Js.argmin(), Js.shape)
    log2_sigma, log2_lmbda = log2_sigmas[min_idx[0]], log2_lambdas[min_idx[1]]
    plt.plot([log2_lmbda], [log2_sigma], "m*", markersize=20)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
