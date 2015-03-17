from kmc.score_matching.random_feats.gaussian_rkhs import xvalidate, \
    sample_basis
from kmc.tools.Log import logger
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def select_sigma_grid(Z, m, num_folds=5, num_repetitions=3,
                        log2_sigma_min=-3, log2_sigma_max=10, resolution_sigma=25,
                        lmbda=0.0001, plot_surface=False):
    
    D = Z.shape[1]
    
    sigmas = 2 ** np.linspace(log2_sigma_min, log2_sigma_max, resolution_sigma)
    
    Js = np.zeros(len(sigmas))
    for i, sigma in enumerate(sigmas):
        gamma = 0.5 * (sigma ** 2)
        
        folds = np.zeros(num_repetitions)
        for j in range(num_repetitions):
            # re-sample basis every repetition
            omega, u = sample_basis(D, m, gamma)
            folds[j] = xvalidate(Z, lmbda, omega, u,
                               n_folds=num_folds, num_repetitions=1)
        
        Js[i] = np.mean(folds)
        logger.info("fold %d/%d, sigma: %.2f, lambda: %.2f, J=%.3f" % \
            (i + 1, len(sigmas), sigma, lmbda, Js[i]))
    
    if plot_surface:
        plt.figure()
        plt.plot(np.log2(sigmas), Js)
    
    return sigmas[Js.argmin()]

def select_sigma_scipy(Z, m, num_folds=5, tol=0.2, num_repetitions=3, lmbda=0.0001):
    D = Z.shape[1]
    def _f(log2_sigma):
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
    
    
    result = sp.optimize.minimize_scalar(_f, tol=tol)
    logger.info("Best sigma: %.2f with value of J=%.3f after %d iterations in %d evaluations" \
                 % (2**result['x'], result['fun'], result['nit'], result['nfev']))
    
    return 2**result['x']