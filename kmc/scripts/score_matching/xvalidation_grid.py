from kmc.score_matching.gaussian_rkhs import xvalidate_sigmas,\
    score_matching_sym
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    sigma = 1.
    lmbda = 1.
    N = 1000
    Z = np.random.randn(N, 2)
    
    num_folds = 10
    num_repetitions = 30
    sigmas = 2**np.linspace(-5, 15, 20)
#     Js = xvalidate_sigmas(Z, sigmas, lmbda,
#                           num_folds=num_folds, num_repetitions=num_repetitions)

    Js = np.zeros(len(sigmas))
    for i,sigma in enumerate(sigmas):
        _, Js[i] = score_matching_sym(Z, sigma, lmbda,
                                                  compute_objective=True)
    
    plt.plot(np.log2(sigmas), Js)
    
    plt.show()
