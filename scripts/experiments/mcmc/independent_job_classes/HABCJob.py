from Crypto.Hash.SHA import SHA1Hash
from abc import abstractmethod
import hashlib
import os

from kmc.densities.gaussian import log_gaussian_pdf
from kmc.hamiltonian.leapfrog import leapfrog_friction_habc_no_storing
from kmc.tools.Log import logger
import numpy as np
from scripts.experiments.mcmc.independent_job_classes.HMCJob import HMCJob, \
    HMCJobResultAggregator
from scripts.tools.SPSA import SPSA


splitted = __file__.split(os.sep)
idx = splitted.index('kamiltonian')
project_path = os.sep.join(splitted[:(idx + 1)])

class DummyHABCTarget(object):
    def __init__(self, abc_target, num_spsa_repeats=1):
                 
        self.abc_target = abc_target
        self.num_spsa_repeats = num_spsa_repeats
        
        # sticky random numbers: fixed seed to simulate data for this instance of DummyHABCTarget
        self.update_fixed_random_state()
        
        # for running average of gradient covariances
        self.grad_cov_est_mean = np.zeros(abc_target.D)
        self.grad_cov_est_M2 = np.zeros((abc_target.D, abc_target.D))
        self.grad_cov_est = np.zeros((abc_target.D, abc_target.D))
        self.grad_cov_est_n = 0
    
    def update_fixed_random_state(self):
        self.fixed_rnd_state = np.random.get_state()
    
    def grad(self, theta):
        logger.debug("Entering")
        
        # update likelihood term
        self._update(theta)
        
        log_lik = lambda theta: log_gaussian_pdf(theta, self.mu, self.L, is_cholesky=True)
        
#         logger.debug("Computing SPSA gradient")
        grad_lik_est = SPSA(log_lik, theta, stepsize=5.,
                                          num_repeats=self.num_spsa_repeats)
        grad_prior = self.abc_target.prior.grad(theta)
        
        
        # update online covariance matrix estimate
        self.grad_cov_est_n += 1
        delta = grad_lik_est - self.grad_cov_est_mean
        self.grad_cov_est_mean += delta/self.grad_cov_est_n
        self.grad_cov_est_M2 += np.outer(delta,grad_lik_est - self.grad_cov_est_mean)
 
        if self.grad_cov_est_n > 1:
            self.grad_cov_est = self.grad_cov_est_M2/(self.grad_cov_est_n - 1)
            logger.debug("Variance grad_0: %.4f" % self.grad_cov_est[0,0])
        
#         logger.debug("grad_lik_est: %s" % str(grad_lik_est))
#         logger.debug("grad_prior: %s" % str(grad_prior))
#         logger.debug("||grad_lik_est||: %.2f" % np.linalg.norm(grad_lik_est))
#         logger.debug("||grad_prior||: %.2f" % np.linalg.norm(grad_prior))
#         logger.debug("||grad_lik_est-grad_prior||: %.2f" % np.linalg.norm(grad_lik_est-grad_prior))
        
        logger.debug("Leaving")
        return grad_lik_est + grad_prior
    
    def _update(self, theta):
        logger.debug("Entering")
        
        state_hash = hashlib.sha1(str(self.fixed_rnd_state)).hexdigest()
        logger.debug("Simulating using rnd_state %s" % state_hash)
        
        D = self.abc_target.D
        
        # sample pseudo data to fit conditional model
        pseudo_datas = np.zeros((self.abc_target.n_lik_samples, D))
        
        # sticky random numbers: fixed seed to simulate data
        current_state = np.random.get_state()
        np.random.set_state(self.fixed_rnd_state)
        for i in range(len(pseudo_datas)):
            pseudo_datas[i] = self.abc_target.simulator(theta)
        np.random.set_state(current_state)
        
        # fit Gaussian, add ridge on diagonal for the epsilon likelihood kernel
        self.mu = np.mean(pseudo_datas, 0)
        Sigma = np.cov(pseudo_datas.T) + np.eye(D) * (self.abc_target.epsilon ** 2)
        self.L = np.linalg.cholesky(Sigma)
        
#         logger.debug("Simulation")
#         logger.debug("Theta: %s" % str(theta[:3]))
#         logger.debug("Mean:  %s" % str(self.mu[:3]))

        logger.debug("Entering")
        
    
class HABCJob(HMCJob):
    def __init__(self, abc_target, momentum,
                 num_iterations, start,
                 num_steps_min=10, num_steps_max=100, step_size_min=0.05,
                 step_size_max=0.3, momentum_seed=0,
                 statistics={}, num_warmup=500, thin_step=1):
        
        HMCJob.__init__(self, abc_target, momentum,
                        num_iterations, start,
                        num_steps_min, num_steps_max, step_size_min,
                        step_size_max, momentum_seed, statistics, num_warmup, thin_step)
        
        self.aggregator = HABCJobResultAggregator()

    @abstractmethod
    def set_up(self):
        # remember orginal abc target for later
        self.abc_target = self.target
        
        # replace with dummy target responsible for gradient computation
        self.target = DummyHABCTarget(self.abc_target)
        
        HMCJob.set_up(self)
        
    @abstractmethod
    def propose(self, current, current_log_pdf, samples, accepted):
        # fixed seed for this trajectory
        self.target.update_fixed_random_state()
        
        # friction leapfrog integrator, broadcasted with current covariance of noise
        c = 1e-5
        V = self.target.grad_cov_est
        self.integrator = lambda q, dlogq, p, dlogp, step_size, num_steps: \
                        leapfrog_friction_habc_no_storing(c, V,
                                                     q, dlogq, p, dlogp, step_size, num_steps)
        
        # use normal HMC mechanics from here
        return HMCJob.propose(self, current, current_log_pdf, samples, accepted)
    
    @abstractmethod
    def accept_prob_log_pdf(self, current, q, p0_log_pdf, p_log_pdf, current_log_pdf=None):
        # potentially re-use log_pdf of last accepted state
        if current_log_pdf is None:
            current_log_pdf = -np.inf
        
        # same as super-class, but with original target
        habc_target = self.target
        self.target = self.abc_target
        
        acc_prob, log_pdf_q = HMCJob.accept_prob_log_pdf(self, current, q, p0_log_pdf, p_log_pdf, current_log_pdf)
        
        # restore target
        self.target = habc_target
    
#         return acc_prob, log_pdf_q
        return 1.0, log_pdf_q
    
    @abstractmethod
    def get_parameter_fname_suffix(self):
        return "HABC_" + HMCJob.get_parameter_fname_suffix(self)[4:] 
    

class HABCJobResultAggregator(HMCJobResultAggregator):
    def __init__(self):
        HMCJobResultAggregator.__init__(self)
        
    @abstractmethod
    def fire_and_forget_result_strings(self):
        strings = HMCJobResultAggregator.fire_and_forget_result_strings(self)
        
        return [str(self.result.D)] + strings

