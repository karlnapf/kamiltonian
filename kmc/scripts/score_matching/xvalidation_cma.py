from kmc.score_matching.gaussian_rkhs_xvalidation import select_sigma_lambda_cma
import numpy as np


if __name__ == "__main__":
    np.random.seed(0)
    N = 200
    Z = np.random.randn(N, 2)
    
    cma_opts={'tolfun':1e-3, 'maxiter':50, 'verb_disp':1}
    
    es = select_sigma_lambda_cma(Z, cma_opts=cma_opts)
    print es.best.get()[0]
