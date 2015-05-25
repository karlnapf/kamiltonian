from kmc.tools.Log import logger
import numpy as np


class ABCPosterior(object):
    def __init__(self, simulator, n_lik_samples, epsilon, prior):
        self.simulator = simulator
        self.n_lik_samples = n_lik_samples
        self.epsilon = epsilon
        self.prior = prior
    
    def log_pdf(self, theta):
        # sample likelihood and evaluate epsilon kernel for each sampled dataset
        liks = np.zeros(self.n_lik_samples)
        logger.debug("Simulating datasets")
        for i in range(self.n_lik_samples):
            pseudo_data = self.simulator(theta)
            
            diff = np.linalg.norm(self.data-pseudo_data)
            within = diff <= self.epsilon
#             logger.debug("Euclidean distance to own data: %.2f", diff)
            liks[i] = 1. if within else 0.
            
#             if liks[i]:
#                 logger.debug("Within epsilon of %.2f", self.epsilon)
        
        m = np.mean(liks)
        logger.debug("Likelihood: %.2f", m)
        return np.log(m) + self.prior(theta)
