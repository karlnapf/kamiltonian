from kmc.score_matching.kernel.kernels import gaussian_kernel
from kmc.score_matching.random_feats.gaussian_rkhs import xvalidate, \
    sample_basis
from kmc.tools.Log import logger
import matplotlib.pyplot as plt
import numpy as np


def select_sigma_grid(Z, m=100, num_folds=5, num_repetitions=3,
                        log2_sigma_min=-3, log2_sigma_max=10, resolution_sigma=25,
                        lmbda=0.00001, plot_surface=False):
    
    D = Z.shape[1]
    
    sigmas = 2 ** np.linspace(log2_sigma_min, log2_sigma_max, resolution_sigma)
    
    Js = np.zeros(len(sigmas))
    for i, sigma in enumerate(sigmas):
        logger.info("fold %d/%d, sigma: %.2f, lambda: %.2f" % \
            (i + 1, len(sigmas), sigma, lmbda))
        
        gamma = 0.5 * (sigma ** 2)
        omega, u = sample_basis(D, m, gamma)
        
        folds = xvalidate(Z, lmbda, omega, u,
                          n_folds=num_folds, num_repetitions=num_repetitions)
        Js[i] = np.mean(folds)
    
    if plot_surface:
        plt.figure()
        plt.plot(np.log2(sigmas), Js)
    
    return sigmas[Js.argmin()]

def select_sigma_lambda_cma(Z, num_folds=5, num_repetitions=1,
                            sigma0=1.1, lmbda0=1.1,
                            cma_opts={}, disp=False):
    import cma
    
    start = np.log2(np.array([sigma0, lmbda0]))
    
    es = cma.CMAEvolutionStrategy(start, 1., cma_opts)
    while not es.stop():
        if disp:
            es.disp()
        solutions = es.ask()
        
        values = np.zeros(len(solutions))
        for i, (log2_sigma, log2_lmbda) in enumerate(solutions):
            sigma = 2 ** log2_sigma
            lmbda = 2 ** log2_lmbda
            
            logger.info("particle %d/%d, sigma: %.2f, lambda: %.2f" % \
                        (i + 1, len(solutions), sigma, lmbda))
            K = gaussian_kernel(Z, sigma=sigma)
            folds = xvalidate(Z, num_folds, sigma, lmbda, K)
            values[i] = np.mean(folds)
        
        es.tell(solutions, values)
    
    return es

