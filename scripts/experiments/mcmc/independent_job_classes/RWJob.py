from abc import abstractmethod

import numpy as np
from scripts.experiments.mcmc.independent_job_classes.MCMCJob import MCMCJob,\
    MCMCJobResultAggregator, MCMCJobResult


class RWJob(MCMCJob):
    def __init__(self, target, num_iterations, start,
                 sigma_proposal,
                 statistics = {}, num_warmup=500, thin_step=1):
        MCMCJob.__init__(self, num_iterations, len(start), start, statistics,
                         num_warmup, thin_step)
        self.aggregator = RWJobResultAggregator()
        
        self.target = target
        self.sigma_proposal = sigma_proposal
    
    @abstractmethod
    def propose(self, current, current_log_pdf, samples, avg_accept):
        # sample from Gaussian at current point with given covariance
        proposal = np.random.randn(self.D)*self.sigma_proposal + current
        
        if current_log_pdf is None:
            current_log_pdf = self.target.log_pdf(current)
        log_pdf_proposal = self.target.log_pdf(proposal)
        
        acc_prob = np.exp(np.minimum(0., log_pdf_proposal-current_log_pdf))
        
        return proposal, acc_prob, log_pdf_proposal

    @abstractmethod
    def get_parameter_fname_suffix(self):
        return "RW_" + MCMCJob.get_parameter_fname_suffix(self)
    
    @abstractmethod
    def submit_to_aggregator(self):
        job_name = self.get_parameter_fname_suffix()
        result = MCMCJobResult(job_name,
                               self.D, self.samples, self.proposals, self.accepted, self.acc_prob, self.log_pdf,
                               self.time_taken_set_up, self.time_taken_sampling,
                               self.num_iterations, self.num_warmup, self.thin_step, self.posterior_statistics)
        self.aggregator.submit_result(result)
        
class RWJobResultAggregator(MCMCJobResultAggregator):
    def __init__(self):
        MCMCJobResultAggregator.__init__(self)