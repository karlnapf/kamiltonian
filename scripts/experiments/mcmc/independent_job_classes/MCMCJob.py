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
        # remember set up time
        start_time = time()
        self.set_up()
        self.time_taken_set_up = time() - start_time
        
        # sampling time
        start_time = time()
        
        self.samples = np.zeros((self.num_iterations, self.D)) + np.nan
        self.proposals = np.zeros((self.num_iterations, self.D)) + np.nan
        self.accepted = np.zeros(self.num_iterations) + np.nan
        self.acc_prob = np.zeros(self.num_iterations) + np.nan
        self.log_pdf = np.zeros(self.num_iterations) + np.nan
        
        
        current = self.start
        current_log_pdf = None
        logger.info("Starting MCMC in D=%d dimensions" % self.D)
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
        
        self.time_taken_sampling = time() - start_time
        
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

    @staticmethod
    def result_dict_from_file(fname):
        """
        Assumes a file with lots of lines as the one created by
        store_fire_and_forget_result and produces a dictionary with D as key
        arrays with experimental results for each of the R repetitions. This contains
        a few standard values as acceptance probability and the used posterior statistics
        """
        results = np.loadtxt(fname)
        
        result_dict = {}
        for i in range(len(results)):
            D = np.int(results[i, 0])
            result_dict[D] = []
    
        for i in range(len(results)):
            D = np.int(results[i, 0])
            time_taken_set_up = np.int(results[i, 1])
            time_taken_sampling = np.int(results[i, 2])
            accepted = np.float(results[i, 3])
            avg_quantile_error = results[i, 4]
            avg_ess = results[i, 5]
            
            to_add = np.zeros(results.shape[1]-1)
            to_add[0] = time_taken_set_up
            to_add[1] = time_taken_sampling
            to_add[2] = accepted
            to_add[3] = avg_quantile_error
            to_add[4] = avg_ess
            
            result_dict[D] += [to_add]
        
        for k,v in result_dict.items():
            result_dict[k] = np.array(v)
        
        return result_dict


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
        D = self.result.mcmc_job.D
        s = []
        s += [str(D)]
        s += [str(self.result.mcmc_job.time_taken_set_up)]
        s += [str(self.result.mcmc_job.time_taken_sampling)]
        s += [str(np.mean(self.result.mcmc_job.accepted))]

        for v in self.result.mcmc_job.posterior_statistics.values():
            # assumes posterior statistics are scalars
            s += [str(v)]
        
        return s