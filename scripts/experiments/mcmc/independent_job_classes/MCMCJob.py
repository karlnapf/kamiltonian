from abc import abstractmethod
import os
import pickle
from time import time
import uuid

from independent_jobs.aggregators.JobResultAggregator import JobResultAggregator
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.JobResult import JobResult

from kmc.tools.Log import logger
import numpy as np


class MCMCJob(IndependentJob):
    def __init__(self, target,
                 num_iterations, D, start, statistics={}, num_warmup=500, thin_step=1):
        
        IndependentJob.__init__(self, MCMCJobResultAggregator())
        
        self.target = target
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
        
        # running average acceptance prob
        self.avg_accept = 0.
        
        self.recompute_log_pdf = False
        

    @abstractmethod
    def compute(self):
        # set up target if possible
        self.target.set_up()
        
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
            if i > 1:
                log_str = "MCMC iteration %d/%d, current log_pdf: %.6f, avg acceptance=%.3f" % (i + 1, self.num_iterations,
                                                                           np.nan if self.log_pdf[i - 1] is None else self.log_pdf[i - 1],
                                                                           self.avg_accept)
            if ((i + 1) % (self.num_iterations / 10)) == 0:
                logger.info(log_str)
            else:
                logger.debug(log_str)
            
            # generate proposal and acceptance probability
            logger.debug("Performing MCMC step")
            
            self.proposals[i], self.acc_prob[i], log_pdf_proposal = self.propose(current,
                                                                                 current_log_pdf, self.samples[:i],
                                                                                 self.avg_accept)
            
            # accept-reject
            r = np.random.rand()
            self.accepted[i] = r < self.acc_prob[i]
            
            logger.debug("Proposed %s" % str(self.proposals[i]))
            logger.debug("Acceptance prob %.4f" % self.acc_prob[i])
            logger.debug("Accepted: %d" % self.accepted[i])
            
            
            # update running mean according to knuth's stable formula
            self.avg_accept += (self.accepted[i] - self.avg_accept) / (i + 1)
            
            # update state
            logger.debug("Updating chain")
            if self.accepted[i]:
                current = self.proposals[i]
                current_log_pdf = log_pdf_proposal
                
                # marginal sampler: do not re-use recompute log-pdf
                if self.recompute_log_pdf:
                    current_log_pdf = None

            # store sample
            self.samples[i] = current
            self.log_pdf[i] = current_log_pdf
        
        self.time_taken_sampling = time() - start_time
        
        logger.info("Computing %d posterior statistics" % len(self.statistics))
        self.posterior_statistics = {}
        for (k, v) in self.statistics.items():
            logger.info("Computing posterior statistic %s using num_warmup=%d, thin=%d" \
                        % (k, self.num_warmup, self.thin_step))
            inds = np.arange(self.num_warmup, len(self.samples), step=self.thin_step)
            self.posterior_statistics[k] = v(self.samples[inds])
        
        logger.info("Submitting results to aggregator")
        self.submit_to_aggregator()
    
    @abstractmethod
    def submit_to_aggregator(self):
        job_name = self.get_parameter_fname_suffix()
        result = MCMCJobResult(job_name,
                               self.D, self.samples, self.proposals, self.accepted, self.acc_prob, self.log_pdf,
                               self.time_taken_set_up, self.time_taken_sampling,
                               self.num_iterations, self.num_warmup, self.thin_step, self.posterior_statistics)
        self.aggregator.submit_result(result)
    
    @abstractmethod
    def set_up(self):
        pass
    
    @abstractmethod
    def propose(self, current, current_log_pdf, samples):
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
            norm_of_mean = results[i, 6]
            
            to_add = np.zeros(6)
            to_add[0] = time_taken_set_up
            to_add[1] = time_taken_sampling
            to_add[2] = accepted
            to_add[3] = avg_quantile_error
            to_add[4] = avg_ess
            to_add[5] = norm_of_mean
            
            result_dict[D] += [to_add]
        
        for k, v in result_dict.items():
            result_dict[k] = np.array(v)
        
        return result_dict

class MCMCJobResult(JobResult):
    def __init__(self, job_name,
                 D, samples, proposals, accepted, acc_prob, log_pdf,
                 time_taken_set_up, time_taken_sampling,
                 num_iterations, num_warmup, thin_step, posterior_statistics):
        JobResult.__init__(self)
        self.job_name = job_name
        self.D = D
        self.samples = samples
        self.proposals = proposals
        self.accepted = accepted
        self.acc_prob = acc_prob
        self.log_pdf = log_pdf
        self.time_taken_set_up = time_taken_set_up
        self.time_taken_sampling = time_taken_sampling
        self.num_iterations = num_iterations
        self.num_warmup = num_warmup
        self.thin_step = thin_step
        self.posterior_statistics = posterior_statistics

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
        fname = folder + os.sep + self.result.job_name + "_" + job_name + ".csv"
        logger.info("Storing fire and forget result in %s" % fname)
        
        with open(fname, 'w+') as f:
            for s in self.fire_and_forget_result_strings():
                f.write(s + " ")

    @abstractmethod
    def fire_and_forget_result_strings(self):
        D = self.result.D
        s = []
        s += [str(D)]
        s += [str(self.result.time_taken_set_up)]
        s += [str(self.result.time_taken_sampling)]
        s += [str(np.mean(self.result.accepted))]

        for _, v in self.result.posterior_statistics.items():
            # assumes posterior statistics are scalars
            s += [str(v)]
        
        return s

class MCMCJobResultAggregatorStoreHome(MCMCJobResultAggregator):
    def __init__(self, path_to_store):
        
        if len(path_to_store) > 1:
            if path_to_store[-1] != os.sep:
                path_to_store += os.sep
        
        self.path_to_store = path_to_store
        MCMCJobResultAggregator.__init__(self)
    
    @abstractmethod
    def store_fire_and_forget_result(self, folder, job_name):
        uni = unicode(uuid.uuid4())
        fname = "%s_ground_truth_iterations=%d_%s.pkl" % \
            (self.result.job_name, self.result.num_iterations, uni)
        full_fname = self.path_to_store + fname
        
        try:
            os.makedirs(self.path_to_store)
        except Exception:
            pass
        
        with open(full_fname, 'w+') as f:
            logger.info("Storing result under %s" % full_fname)
            pickle.dump(self.result, f)
