from abc import abstractmethod
import os
from time import time

from independent_jobs.aggregators.JobResultAggregator import JobResultAggregator
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.JobResult import JobResult

from kmc.tools.Log import logger
import numpy as np

class MCMCJob(IndependentJob):
    def __init__(self,
                 num_iterations, D, start, statistics = {}, num_warmup=500, thin_step=1):
        
        IndependentJob.__init__(self, MCMCJobResultAggregator())
        
        self.num_iterations = num_iterations
        self.D = D
        self.start = start
        self.statistics = statistics
        self.num_warmup = num_warmup
        self.thin_step = thin_step
        
        assert len(start.shape) == 1
        assert len(start) == D
        
        # short queue
        self.walltime = 1 * 60 * 60
        

    @abstractmethod
    def compute(self):
        self.set_up()
        
        start_time = time()
        
        self.samples = np.zeros((self.num_iterations, self.D)) + np.nan
        self.proposals = np.zeros((self.num_iterations, self.D)) + np.nan
        self.accepted = np.zeros(self.num_iterations) + np.nan
        self.acc_prob = np.zeros(self.num_iterations) + np.nan
        self.log_pdf = np.zeros(self.num_iterations) + np.nan
        
        
        current = self.start
        current_log_pdf = None
        logger.info("Starting MCMC")
        for i in range(self.num_iterations):
            # print chain progress
            log_str = "MCMC iteration %d/%d" % (i + 1, self.num_iterations)
            if ((i+1) % (self.num_iterations / 10)) == 0:
                logger.info(log_str)
            else:
                logger.debug(log_str)
            
            # generate proposal and acceptance probability
            logger.debug("Generating proposal")
            self.proposals[i], self.acc_prob[i], log_pdf_proposal = self.propose(current, current_log_pdf)
            
            # accept-reject
            logger.debug("Updating chain")
            r = np.random.rand()
            self.accepted[i] = r < self.acc_prob[i]
            
            # update state
            if self.accepted[i]:
                current = self.proposals[i]
                current_log_pdf = log_pdf_proposal

            # store sample
            self.samples[i] = current
            self.log_pdf[i] = current_log_pdf
        
        self.time_taken = time() - start_time
        
        logger.info("Computing %d posterior statistics" % len(self.statistics))
        self.posterior_statistics = {}
        for (k,v) in self.statistics.items():
            logger.info("Computing posterior statistic %s using num_warmup=%d, thin=%d" \
                        % (k, self.num_warmup, self.thin_step))
            inds = np.arange(self.num_warmup, len(self.samples), step=self.thin_step)
            self.posterior_statistics[k] = v(self.samples[inds])
        
        logger.info("Submitting results to aggregator")
        self.submit_to_aggregator()
    
    @abstractmethod
    def submit_to_aggregator(self):
        result = MCMCJobResult(self)
        self.aggregator.submit_result(result)
    
    @abstractmethod
    def set_up(self):
        pass
    
    @abstractmethod
    def propose(self, current):
        raise NotImplementedError()

    @abstractmethod
    def get_parameter_fname_suffix(self):
        return ("D=%d" % self.D)

class MCMCJobResult(JobResult):
    def __init__(self, mcmc_job):
        JobResult.__init__(self)
        self.mcmc_job = mcmc_job


class MCMCJobResultAggregator(JobResultAggregator):
    def __init__(self):
        JobResultAggregator.__init__(self, 1)
        
    def finalize(self):
        pass
    
    def submit_result(self, result):
        self.result = result
    
    def get_final_result(self):
        return self.result
    
    def clean_up(self):
        pass
    
    def store_fire_and_forget_result(self, folder, job_name):
        fname = folder + os.sep + self.result.mcmc_job.get_parameter_fname_suffix() + "_" + job_name + ".csv"
        logger.info("Storing fire and forget result in %s" % fname)
        
        with open(fname, 'w+') as f:
            for s in self.fire_and_forget_result_strings():
                f.write(s + " ")

    @abstractmethod
    def fire_and_forget_result_strings(self):
        s = []
        D = self.result.mcmc_job.D
        for _, v in self.result.mcmc_job.posterior_statistics.items():
            # assumes posterior statistics are vectors
            s += [" ".join([str(D)] + [str(v[i]) for i in range(len(v))] + [str(np.mean(self.result.mcmc_job.accepted))])]
        
        return s