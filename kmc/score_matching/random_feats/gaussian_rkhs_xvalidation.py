from multiprocessing.pool import Pool

from kmc.score_matching.random_feats.gaussian_rkhs import xvalidate, \
    sample_basis
from kmc.tools.Log import logger
# import matplotlib.pyplot as plt
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
    
#     if plot_surface:
#         plt.figure()
#         plt.plot(np.log2(sigmas), Js)
    
    return sigmas[Js.argmin()]

def select_sigma_scipy(Z, m, num_folds=5, tol=0.2, num_repetitions=3, lmbda=0.0001):
    D = Z.shape[1]
    def objective(log2_sigma):
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
    
    
    result = sp.optimize.minimize_scalar(objective, tol=tol)
    logger.info("Best sigma: %.2f with value of J=%.3f after %d iterations in %d evaluations" \
                 % (2 ** result['x'], result['fun'], result['nit'], result['nfev']))
    
    return 2 ** result['x']

def multicore_fun(log2_sigma, log2_lmbda, num_repetitions, num_folds, Z, m):
    D = Z.shape[1]
    
    lmbda = 2 ** log2_lmbda
    sigma = 2 ** log2_sigma
    gamma = 0.5 * (sigma ** 2)
    
    folds = np.zeros(num_repetitions)
    for j in range(num_repetitions):
        logger.debug("xvalidation repetition %d/%d" % (j+1, num_repetitions))
        omega, u = sample_basis(D, m, gamma)
        folds[j] = np.mean(xvalidate(Z, lmbda, omega, u,
                                     num_folds, num_repetitions))
    
    result = np.mean(folds)
    logger.info("particle, sigma: %.2f, lambda: %.6f, J=%.4f" % \
        (sigma, lmbda, result))
    return result

def multicore_fun_helper(args):
    return multicore_fun(*args)

def select_sigma_lambda_cma(Z, m, num_threads=6, num_folds=5, num_repetitions=1,
                            sigma0=0.5, lmbda0=0.0001,
                            cma_opts={}, disp=False):
    import cma
    
    start = np.log2(np.array([sigma0, lmbda0]))
    
    pool = Pool(num_threads)
    chunksize = 1
    
    es = cma.CMAEvolutionStrategy(start, 1., cma_opts)
    while not es.stop():
        if disp:
            es.disp()
        solutions = es.ask()
        
        # use multicore here
        args = [(log2_sigma, log2_lmbda, num_repetitions, num_folds, Z, m) for (log2_sigma, log2_lmbda) in solutions]
        values = np.zeros(len(solutions))
        for i, result in enumerate(pool.imap(multicore_fun_helper, args), chunksize):
            values.flat[i - 1] = result
        
        es.tell(solutions, values)
    
    sigma = 2 ** es.best.get()[0][0]
    lmbda = 2 ** es.best.get()[0][1]
    
    return sigma, lmbda
