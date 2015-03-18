from kmc.score_matching.gaussian_rkhs import xvalidate
from kmc.score_matching.gaussian_rkhs_xvalidation import select_sigma_lambda_cma
from kmc.score_matching.kernel.kernels import gaussian_kernel
import numpy as np


if __name__ == "__main__":
    np.random.seed(0)
    N = 200
    Z = np.random.randn(N, 2)
#     print np.sum(Z) * np.std(Z) * np.sum(Z**2) * np.std(Z**2)
    
    cma_opts = {'tolfun':1e-2, 'maxiter':20, 'verb_disp':1,
                'bounds': [-3, 5]}
    
    es = select_sigma_lambda_cma(Z, cma_opts=cma_opts, disp=True)
    log2_sigma = es.best.get()[0][0]
    log2_lmbda = es.best.get()[0][1]
    sigma = 2 ** log2_sigma
    lmbda = 2 ** log2_lmbda
    K = gaussian_kernel(Z, sigma=sigma)
    print log2_sigma, log2_lmbda, np.mean(xvalidate(Z, 5, sigma, lmbda, K))
