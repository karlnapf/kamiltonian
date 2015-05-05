from abc import abstractmethod
import os

from kmc.score_matching.kernel.kernels import gaussian_kernel,\
    gaussian_kernel_grad
from kmc.score_matching.random_feats.estimator import RandomFeatsEstimator
from kmc.score_matching.random_feats.gaussian_rkhs import sample_basis, \
    score_matching_sym
from kmc.score_matching.random_feats.gaussian_rkhs_xvalidation import select_sigma_lambda_cma
from kmc.tools.Log import logger
import numpy as np
from scripts.experiments.mcmc.independent_job_classes.HMCJob import HMCJob, \
    HMCJobResultAggregator
from scripts.experiments.mcmc.independent_job_classes.KMCRandomFeatsJob import KMCJobResultAggregator
from scripts.experiments.mcmc.independent_job_classes.MCMCJob import MCMCJob,\
    MCMCJobResultAggregator
from kmc.densities.gaussian import sample_gaussian


splitted = __file__.split(os.sep)
idx = splitted.index('kamiltonian')
project_path = os.sep.join(splitted[:(idx + 1)])

class KameleonJob(MCMCJob):
    def __init__(self, Z, sigma, nu2, gamma2,
                 target, 
                 num_iterations, start,
                 statistics={}, num_warmup=500, thin_step=1):
        
        MCMCJob.__init__(self, num_iterations, Z.shape[1], start, statistics, num_warmup, thin_step)
        
        self.aggregator = KMCJobResultAggregator()
        
        self.target = target
        self.Z = Z
        self.sigma = sigma
        self.nu2 = nu2
        self.gamma2 = gamma2
        
    @abstractmethod
    def set_up(self):
        logger.info("Using sigma=%.2f, nu2=%.2f, gamma2=%.2f" % \
                    (self.sigma, self.nu2, self.gamma2))
        
        MCMCJob.set_up(self)
        
    def compute_proposal_covariance(self, y):
        """
        Pre-computes constants of the log density of the proposal distribution,
        which is Gaussian as p(x|y) ~ N(mu, R)
        where
        mu = y-a
        a = 0
        R  = gamma^2 I + M M^T
        M  = 2 [\nabla_x k(x,z_i]|_x=y
        
        Returns (mu,L_R), where L_R is lower Cholesky factor of R
        """
        assert(len(np.shape(y)) == 1)
        
        # M = 2 [\nabla_x k(x,z_i]|_x=y
        R = self.gamma2 * np.eye(len(y))
        if self.Z is not None:
            M = 2 * gaussian_kernel_grad(y, self.Z, sigma=self.sigma)
            # R = gamma^2 I + \nu^2 * M H M^T
            H = np.eye(len(self.Z)) - 1.0 / len(self.Z)
            R += self.nu2 * M.T.dot(H.dot(M))
            
        L_R = np.linalg.cholesky(R)
        
        return L_R
    
    @abstractmethod
    def propose(self, current, current_log_pdf=None):
        # compute Kameleon proposal
        L = self.compute_proposal_covariance(current)
        
        # sample from Gaussian at current point with given covariance
        proposal = sample_gaussian(1, current, L, is_cholesky=True)[0]
        
        if current_log_pdf is None:
            current_log_pdf = self.target.log_pdf(current)
        log_pdf_proposal = self.target.log_pdf(proposal)
        
        acc_prob = np.exp(np.minimum(0., log_pdf_proposal-current_log_pdf))
        
        return proposal, acc_prob, log_pdf_proposal
    
    @abstractmethod
    def get_parameter_fname_suffix(self):
        return ("Kameleon_N=%d_" % len(self.Z)) + MCMCJob.get_parameter_fname_suffix(self)[4:] 
    
class KameleonJobResultAggregator(MCMCJobResultAggregator):
    def __init__(self):
        MCMCJobResultAggregator.__init__(self)
        
    @abstractmethod
    def fire_and_forget_result_strings(self):
        strings = MCMCJobResultAggregator.fire_and_forget_result_strings(self)
        
        return [str(len(self.result.mcmc_job.Z))] + strings