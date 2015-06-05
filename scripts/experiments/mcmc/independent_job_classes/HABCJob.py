from abc import abstractmethod
from numpy.testing.utils import assert_allclose
import os

from kmc.densities.gaussian import log_gaussian_pdf
from kmc.tools.Log import logger
import numpy as np
from scripts.experiments.mcmc.independent_job_classes.HMCJob import HMCJob, \
    HMCJobResultAggregator


splitted = __file__.split(os.sep)
idx = splitted.index('kamiltonian')
project_path = os.sep.join(splitted[:(idx + 1)])

def SPSA(loss, theta, stepsize, num_repeats=1):
    """
    Implements Simultaneous perturbation stochastic approximation to estimate
    gradient of given loss function
    
    """
    D = len(theta)
    grad_ests = np.zeros((num_repeats, D))
    
    for i in range(num_repeats):
        delta = 2 * (np.random.rand(D) > .5).astype(int) - 1
        thetaplus = theta + stepsize * delta;
        thetaminus = theta - stepsize * delta
        yplus = loss(thetaplus)
        yminus = loss(thetaminus);
        grad_ests[i] = (yplus - yminus) / (2 * stepsize * delta)
    
#     L=np.linalg.cholesky(np.cov(grad_ests.T) + np.eye(D)*1e-5)
#     print 2*np.sum(np.log(np.diag(L)))
    
    return np.mean(grad_ests, 0)

def test_SPSA():
    loss = lambda x: 0.5 * x.dot(x)
    grad_true = lambda x: x
    
    x_test = np.array([1, 2])
    grad_est = SPSA(loss, x_test, stepsize=.05, num_repeats=1000)
    assert_allclose(grad_true(x_test), grad_est, atol=.1)
    
if __name__ == "__main__":
    test_SPSA()
    

class DummyHABCTarget(object):
    def __init__(self, abc_target, num_spsa_repeats=1):
                 
        self.abc_target = abc_target
        
        # sticky random numbers: fixed seed to simulate data for this instance of DummyHABCTarget
        self.fixed_rnd_state = np.random.get_state()
        self.num_spsa_repeats = num_spsa_repeats
#         self.current_theta = None
    
    def grad(self, theta):
#         logger.info("theta_0=%.2f" % theta[0])
        
        # update likelihood term
        self._update(theta)
        
        log_lik = lambda theta: log_gaussian_pdf(theta, self.mu, self.L, is_cholesky=True)
        
#         logger.debug("Computing SPSA gradient")
        grad_lik_est = SPSA(log_lik, theta, stepsize=5., num_repeats=self.num_spsa_repeats)
        grad_prior = self.abc_target.prior.grad(theta)
        
#         logger.debug("grad_lik_est: %s" % str(grad_lik_est))
#         logger.debug("grad_prior: %s" % str(grad_prior))
#         logger.debug("||grad_lik_est||: %.2f" % np.linalg.norm(grad_lik_est))
#         logger.debug("||grad_prior||: %.2f" % np.linalg.norm(grad_prior))
#         logger.debug("||grad_lik_est-grad_prior||: %.2f" % np.linalg.norm(grad_lik_est-grad_prior))
        
        return grad_lik_est + grad_prior
    
    def _update(self, theta):
#         # dont re-simulate if not needed
#         if self.current_theta is not None and np.allclose(self.current_theta, theta):
#             logger.debug("Theta=%s was already simulated from." % str(theta))
#             return
#         else:
#             self.current_theta = theta
        
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
