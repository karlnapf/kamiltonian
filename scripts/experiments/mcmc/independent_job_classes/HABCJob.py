from abc import abstractmethod
import os

from kmc.densities.gaussian import log_gaussian_pdf
import numpy as np
from scripts.experiments.mcmc.independent_job_classes.HMCJob import HMCJob, \
    HMCJobResultAggregator


splitted = __file__.split(os.sep)
idx = splitted.index('kamiltonian')
project_path = os.sep.join(splitted[:(idx + 1)])

temp = 0

def SPSA(loss, theta, stepsize):
    """
    Implements Simultaneous perturbation stochastic approximation to estimate
    gradient of given loss function
    
    """
    D = len(theta)
    delta=2*(np.random.rand(D)>.5).astype(int)-1
    thetaplus=theta+stepsize*delta;
    thetaminus=theta-stepsize*delta
    yplus=loss(thetaplus)
    yminus=loss(thetaminus);
    grad_est=(yplus-yminus)/(2*stepsize*delta)
    
    return grad_est

class DummyHABCTarget(object):
    def __init__(self, abc_target):
                 
        self.abc_target = abc_target
    
    def grad(self, theta):
#         logger.debug("Computing SPSA gradient")
        g_est = SPSA(self.log_lik, theta, stepsize=.05)
        
        return g_est + self.abc_target.prior.grad(theta)
    
    def update(self, theta):
        D = self.abc_target.D
        
        # sample pseudo data to fit conditional model
        pseudo_datas = np.zeros((self.abc_target.n_lik_samples, D))
        for i in range(len(pseudo_datas)):
            pseudo_datas[i] = self.abc_target.simulator(theta)
        
        # fit Gaussian, add ridge on diagonal for the epsilon likelihood kernel
        self.mu = np.mean(pseudo_datas, 0)
        Sigma = np.cov(pseudo_datas) + np.eye(D)*(self.abc_target.epsilon**2)
        self.L = np.linalg.cholesky(Sigma)
    
    def log_lik(self, theta):
        self.update(theta)
        return log_gaussian_pdf(theta, self.mu, self.L, is_cholesky=True)
    
    def log_pdf(self, theta):
        self.update(theta)
        log_lik = self.log_lik(theta)
            
        return log_lik  + self.abc_target.prior.log_pdf(theta)

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
        
        HMCJob.set_up(self)

    @abstractmethod
    def propose(self, current, current_log_pdf, samples, accepted):
        # replace with dummy target responsible for gradient computation
        self.target = DummyHABCTarget(self.abc_target)
        
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
