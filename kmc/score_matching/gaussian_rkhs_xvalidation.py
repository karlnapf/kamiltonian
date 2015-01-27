from kmc.score_matching.gaussian_rkhs import xvalidate
from kmc.score_matching.kernel.kernels import gaussian_kernel
import numpy as np


def select_sigma_lambda_grid(Z, num_folds=5, num_repetitions=1,
                        log2_sigma_min=-5, log2_sigma_max=5, resolution_sigma=10,
                        log2_lambda_min=-5, log2_lambda_max=5, resolution_lambda=10,
                        ):
    sigmas = 2 ** np.linspace(log2_sigma_min, log2_sigma_max, resolution_sigma)
    lambdas = 2 ** np.linspace(log2_lambda_min, log2_lambda_max, resolution_lambda)

    Js = np.zeros((len(sigmas), len(lambdas)))
    for i, sigma in enumerate(sigmas):
        for j, lmbda in enumerate(lambdas):
            print "sigma: %.2f, lambda: %.2f" % (sigma, lmbda)
            K = gaussian_kernel(Z, sigma=sigma)
            folds = xvalidate(Z, num_folds, sigma, lmbda, K)
            Js[i, j] = np.mean(folds)
    
    min_idx = np.unravel_index(Js.argmin(), Js.shape)
    return sigmas[min_idx[0]], lambdas[min_idx[1]]

def select_sigma_lambda_cma(Z, num_folds=5, num_repetitions=1,
                            sigma0=1., lmbda0=1.,
                            cma_opts={}):
    import cma
    
    start = np.log2(np.array([sigma0, lmbda0]))
    
    es = cma.CMAEvolutionStrategy(start, 1., cma_opts)
    while not es.stop():
        es.disp()
        solutions = es.ask()
        
        values = np.zeros(len(solutions))
        for i, (log2_sigma, log2_lmbda) in enumerate(solutions):
            sigma = 2 ** log2_sigma
            lmbda = 2 ** log2_lmbda
            
#             print "log2_sigma: %.2f, log2_lmbda: %.2f" % (log2_sigma, log2_lmbda) 
            K = gaussian_kernel(Z, sigma=sigma)
            folds = xvalidate(Z, num_folds, sigma, lmbda, K)
            values[i] = np.mean(folds)
        
        es.tell(solutions, values)
    
    return es

