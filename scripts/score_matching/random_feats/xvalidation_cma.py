from kmc.score_matching.random_feats.gaussian_rkhs import xvalidate, \
    sample_basis
from kmc.score_matching.random_feats.gaussian_rkhs_xvalidation import select_sigma_lambda_cma
import numpy as np


if __name__ == "__main__":
    np.random.seed(0)
    N = 20
    D = 2
    Z = np.random.randn(N, D)
    m = N
#     print np.sum(Z) * np.std(Z) * np.sum(Z**2) * np.std(Z**2)
    num_folds = 5
    num_repetitions = 3

    
    cma_opts = {'tolfun':0.1, 'maxiter':1, 'verb_disp':1}
    num_threads = 2
    
    # D = 2
    sigma0 = 0.51
    lmbda0 = 0.0000081
    
    sigma, lmbda, es = select_sigma_lambda_cma(Z, m, num_threads, num_folds, num_repetitions,
                                 sigma0, lmbda0, cma_opts, return_cma=True)
    gamma = 0.5 * (sigma ** 2)

    omega, u = sample_basis(D, m, gamma)
    
    print(sigma, lmbda,
          np.mean(xvalidate(Z, lmbda, omega, u, n_folds=5, num_repetitions=3)))
