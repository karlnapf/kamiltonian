from kmc.tools.Log import logger
import numpy as np


class ABCPosterior(object):
    def __init__(self, simulator, n_lik_samples, epsilon, prior):
        self.simulator = simulator
        self.n_lik_samples = n_lik_samples
        self.epsilon = epsilon
        self.prior = prior
    
    def log_pdf(self, theta):
        # sample likelihood and evaluate Gaussian epsilon kernel for each sampled dataset
        log_liks = np.zeros(self.n_lik_samples)
        logger.debug("Simulating datasets")
        for i in range(self.n_lik_samples):
            # summary statistic: mean
            pseudo_data = np.mean(self.simulator(theta), 0)
            
            diff = np.linalg.norm(pseudo_data-self.data)
#             logger.debug("Diff=%.6f" % diff)
            log_liks[i] = -0.5 * (diff**2) / self.epsilon**2
            
        m = np.mean(np.exp(log_liks))
        logger.debug("Likelihood: %.2f", m)
        
        result = np.log(m) + self.prior.log_pdf(theta)
        
        if np.isnan(result):
            result = -np.inf
        
        return result
