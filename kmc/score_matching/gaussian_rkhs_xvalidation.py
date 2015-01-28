from kmc.score_matching.gaussian_rkhs import xvalidate
from kmc.score_matching.kernel.kernels import gaussian_kernel
import numpy as np
import matplotlib.pyplot as plt

def select_sigma_grid(Z, num_folds=5, num_repetitions=1,
                        log2_sigma_min=-3, log2_sigma_max=10, resolution_sigma=25,
                        lmbda=1., plot_surface=False):
    
    sigmas = 2 ** np.linspace(log2_sigma_min, log2_sigma_max, resolution_sigma)

    Js = np.zeros(len(sigmas))
    for i, sigma in enumerate(sigmas):
#             print "sigma: %.2f, lambda: %.2f" % (sigma, lmbda)
            K = gaussian_kernel(Z, sigma=sigma)
            folds = xvalidate(Z, num_folds, sigma, lmbda, K)
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
        for i, (log2_sigma,log2_lmbda) in enumerate(solutions):
            sigma = 2 ** log2_sigma
            lmbda = 2 ** log2_lmbda
            
#             print "sigma: %.2f, lambda: %.2f" % (sigma, lmbda)
            
#             print "log2_sigma: %.2f, log2_lmbda: %.2f" % (log2_sigma, log2_lmbda) 
            K = gaussian_kernel(Z, sigma=sigma)
            folds = xvalidate(Z, num_folds, sigma, lmbda, K)
            values[i] = np.mean(folds)
        
        es.tell(solutions, values)
    
    return es

