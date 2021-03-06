from abc import abstractmethod

from kmc.hamiltonian.leapfrog import leapfrog_no_storing
from kmc.tools.Log import logger
import numpy as np
from scripts.experiments.mcmc.independent_job_classes.MCMCJob import MCMCJob,\
    MCMCJobResultAggregator


class HMCJob(MCMCJob):
    def __init__(self, target, momentum,
                 num_iterations, start,
                 num_steps_min=10, num_steps_max=100, step_size_min=0.05,
                 step_size_max=0.3, momentum_seed=0,
                 statistics = {}, num_warmup=500, thin_step=1):
        MCMCJob.__init__(self, target, num_iterations, momentum.D, start, statistics,
                         num_warmup, thin_step)
        self.aggregator = HMCJobResultAggregator()
        
        self.momentum = momentum
        self.num_steps_min = num_steps_min
        self.num_steps_max = num_steps_max
        self.step_size_min = step_size_min
        self.step_size_max = step_size_max
        self.momentum_seed = momentum_seed
    
    @abstractmethod
    def set_up(self):
        # momentum is fixed so sample all of them here
        logger.info("Using momentum seed: %d" % self.momentum_seed)
        np.random.seed(self.momentum_seed)
        self.hmc_rnd_state = np.random.get_state()

        # standard leapfrog integrator
        self.integrator = leapfrog_no_storing

        MCMCJob.set_up(self)
    
    @abstractmethod
    def propose(self, current, current_log_pdf, samples, accepted):
        # random variables from a fixed random stream without modifying the current one
        rnd_state = np.random.get_state()
        np.random.set_state(self.hmc_rnd_state)
        
        # sample momentum and leapfrog parameters
        p0 = self.momentum.sample()
        p0_log_pdf = self.momentum.log_pdf(p0)
        num_steps = np.random.randint(self.num_steps_min, self.num_steps_max + 1)
        step_size = np.random.rand() * (self.step_size_max - self.step_size_min) + self.step_size_min
        
        # restore random state
        self.hmc_rnd_state = np.random.get_state()
        np.random.set_state(rnd_state)
        
        logger.debug("Simulating Hamiltonian flow")
        q, p = self.integrator(current, self.target.grad, p0, self.momentum.grad, step_size, num_steps)
        
        # compute acceptance probability, extracting log_pdf of q
        p_log_pdf = self.momentum.log_pdf(p)
        
        # use a function call to be able to overload it for KMC
        acc_prob, log_pdf_q = self.accept_prob_log_pdf(current, q, p0_log_pdf, p_log_pdf, current_log_pdf)
        
        return q, acc_prob, log_pdf_q

    @abstractmethod
    def accept_prob_log_pdf(self, current, q, p0_log_pdf, p_log_pdf, current_log_pdf=None):
        # potentially re-use log_pdf of last accepted state
        if current_log_pdf is None:
            current_log_pdf = self.target.log_pdf(current)
        
        log_pdf_q = self.target.log_pdf(q)
        H0 = -current_log_pdf - p0_log_pdf
        H = -log_pdf_q - p_log_pdf
        difference = -H + H0
        acc_prob = np.exp(np.minimum(0., difference))
        
        logger.debug("log_pdf current=%.2f" % current_log_pdf)
        logger.debug("log_pdf q=%.2f" % log_pdf_q)
        logger.debug("difference_q=%.2f" % (log_pdf_q-current_log_pdf))
        logger.debug("difference_p=%.2f" % (p_log_pdf-p0_log_pdf))
        logger.debug("H0=%.2f" % H0)
        logger.debug("H=%.2f" % H)
        logger.debug("H0-H=%.2f" % difference)
        
        return acc_prob, log_pdf_q
    
    @abstractmethod
    def get_parameter_fname_suffix(self):
        return "HMC_" + MCMCJob.get_parameter_fname_suffix(self)
    
class HMCJobResultAggregator(MCMCJobResultAggregator):
    def __init__(self):
        MCMCJobResultAggregator.__init__(self)