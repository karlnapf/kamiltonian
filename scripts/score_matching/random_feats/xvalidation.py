from kmc.score_matching.random_feats.gaussian_rkhs import xvalidate, \
    sample_basis
from kmc.tools.Log import logger
import numpy as np
import scipy as sp


if __name__ == '__main__':
    N = 200
    D = 2
    Z = np.random.randn(N, D)
    m = N
    num_folds = 5
    num_repetitions = 10
    lmbda = 0.001

    def sigma_objective(log2_sigma, lmbda):
            sigma = 2 ** log2_sigma
            folds = np.zeros(num_repetitions)
            for i in range(num_repetitions):
                gamma = 0.5 * (sigma ** 2)
                omega, u = sample_basis(D, m, gamma)
                folds[i] = np.mean(xvalidate(Z, lmbda, omega, u, num_folds, num_repetitions=1))
            
            result = np.mean(folds)
            logger.info("xvalidation, sigma: %.2f, lambda: %.2f, J=%.3f" % \
                        (sigma, lmbda, result))
            return result
    
    def lmbda_objective(log2_lmbda, sigma):
        lmbda = 2**log2_lmbda
        
        gamma = 0.5 * (sigma ** 2)
        omega, u = sample_basis(D, m, gamma)
        
        folds = xvalidate(Z, lmbda, omega, u, num_folds, num_repetitions=num_repetitions)
        result = np.mean(folds)
        logger.info("xvalidation, sigma: %.2f, lambda: %.2f, J=%.3f" % \
                    (sigma, lmbda, result))
        return result
    
    log2_lmbda = 0.
    sigma = 0.8
    print(lmbda_objective(log2_lmbda, sigma))
    
