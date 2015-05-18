from kmc.score_matching.kernel.kernels import gaussian_kernel,\
    gaussian_kernel_grad
import numpy as np


def log_pdf_estimate(x, alpha, Z, kernel):
    k = kernel(Z, x[np.newaxis,:])[:, 0]
    return alpha.dot(k)

def log_pdf_estimate_grad(x, alpha, Z, kernel_grad):
    k = kernel_grad(x, Z)
    return alpha.dot(k)

class LiteEstimatorGaussian(object):
    def __init__(self, alpha, Z, sigma):
        self.alpha = alpha
        self.Z = Z
        self.sigma = sigma
        
    def log_pdf(self, x):
        k = gaussian_kernel(self.Z, x[np.newaxis,:], sigma=self.sigma)[:, 0]
        return self.alpha.dot(k)
    
    def grad(self, x):
        k = gaussian_kernel_grad(x, self.Z, sigma=self.sigma)
        return self.alpha.dot(k)

