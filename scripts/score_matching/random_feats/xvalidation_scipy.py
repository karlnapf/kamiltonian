from kmc.score_matching.random_feats.gaussian_rkhs import xvalidate, \
    sample_basis
from kmc.tools.Log import logger
import numpy as np
import scipy as sp


if __name__ == '__main__':
    N = 200
    D = 20
    Z = np.random.randn(N, D)
    m = N
    num_folds = 5
    num_repetitions = 10
    
    #D = 10
    lmbda = 0.000250
    sigma = 0.60

    def sigma_objective(log2_sigma, lmbda=lmbda):
            sigma = 2 ** log2_sigma
            folds = np.zeros(num_repetitions)
            for i in range(num_repetitions):
                gamma = 0.5 * (sigma ** 2)
                omega, u = sample_basis(D, m, gamma)
                folds[i] = np.mean(xvalidate(Z, lmbda, omega, u, num_folds, num_repetitions=1))
            
            result = np.mean(folds)
            logger.info("xvalidation, sigma: %.2f, lambda: %.6f, J=%.3f" % \
                        (sigma, lmbda, result))
            return result
    
    def lmbda_objective(log2_lmbda, sigma=sigma):
        lmbda = 2**log2_lmbda
        
        gamma = 0.5 * (sigma ** 2)
        omega, u = sample_basis(D, m, gamma)
        
        folds = xvalidate(Z, lmbda, omega, u, num_folds, num_repetitions=num_repetitions)
        result = np.mean(folds)
        logger.info("xvalidation, sigma: %.2f, lambda: %.6f, J=%.4f" % \
                    (sigma, lmbda, result))
        return result
    
    
    while True:
        f = lambda log2_sigma: sigma_objective(log2_sigma, lmbda)
        result = sp.optimize.minimize_scalar(f,
                                             bracket=[np.log2(sigma)-1, np.log2(sigma)+1],
                                             tol=0.2)
        print(result)
        sigma = 2**result['x']
        print("sigma: %.2f, lambda: %.6f" % (sigma, lmbda))
        
        g = lambda log2_lmbda: lmbda_objective(log2_lmbda, sigma)
        result = sp.optimize.minimize_scalar(g,
                                             bracket=[np.log2(lmbda)-1, np.log2(lmbda)+1],
                                             tol=0.2)
        print(result)
        lmbda = 2**result['x']
        print("sigma: %.2f, lambda: %.6f" % (sigma, lmbda))
