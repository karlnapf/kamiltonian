import modshogun as sg

import numpy as np
from kmc.tools.Log import logger
from kmc.densities.gaussian import log_gaussian_pdf

class PseudoMarginalHyperparameters(object):
    """
    Class to represent a GP's marginal posterior distribution of hyperparameters
    
    p(theta|y) \propto p(y|theta) p(theta)
    
    as an MCMC target. The p(y|theta) function is an unbiased estimate.
    Hyperparameters are the length scales of a Gaussian ARD kernel.
    
    Uses the Shogun machine learning toolbox for GP inference.
    """
    def __init__(self, X, y, n_importance, prior_log_pdf, ridge=0., num_shogun_threads=1):
        self.n_importance=n_importance
        self.prior_log_pdf=prior_log_pdf
        self.ridge=ridge
        self.X=X
        self.y=y
        
        # tell shogun to use 1 thread only
        logger.debug("Using Shogun with %d threads" % num_shogun_threads)
        sg.ZeroMean().parallel.set_num_threads(num_shogun_threads)
    
        # shogun representation of data
        self.sg_labels=sg.BinaryLabels(self.y)
        self.sg_feats_train=sg.RealFeatures(self.X.T)
        
        # ARD: set set theta, which is in log-scale, as kernel weights
        self.sg_kernel=sg.GaussianARDKernel(10,1)
        
        self.sg_mean=sg.ZeroMean()
        self.sg_likelihood=sg.LogitLikelihood()
        
    def log_pdf(self, theta):
        self.sg_kernel.set_vector_weights(np.exp(theta))
        inference=sg.EPInferenceMethod(
#         inference=sg.SingleLaplacianInferenceMethod(
                                        self.sg_kernel,
                                        self.sg_feats_train,
                                        self.sg_mean,
                                        self.sg_labels,
                                        self.sg_likelihood)

        # fix kernel scaling for now
        inference.set_scale(1.)
        
        log_ml_estimate=inference.get_marginal_likelihood_estimate(self.n_importance, self.ridge)
        
        # prior is also in log-domain, so no exp of theta
        log_prior=self.prior_log_pdf(theta)
        result=log_ml_estimate+log_prior
            
        return result
    

def prior_log_pdf(x):
    D= len(x)
    return log_gaussian_pdf(x, mu=0.*np.ones(D), Sigma=np.eye(D)*5)

if __name__=="__main__":
    np.random.seed(0)
    N = 10
    D = 2
    X = np.random.randn(N, D)
    y = (X[:,0]>0).astype(np.float64)
    n_importance = 100
    prior = prior_log_pdf
    ridge = 1e-3
    pm = PseudoMarginalHyperparameters(X, y, n_importance, prior, ridge)
    
    theta = np.random.randn(D)
    theta = np.zeros(D)
    
    print pm.log_pdf(theta)
