

from abc import abstractmethod
import os

from independent_jobs.aggregators.JobResultAggregator import JobResultAggregator
import numpy as np

class TrajectoryJobResultAggregator(JobResultAggregator):
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
    
    @abstractmethod
    def store_fire_and_forget_result(self, folder, job_name):
        D = self.result.D
        N = self.result.N
        
        fname = folder + os.sep + "N=%d_D=%d_%s" % (N, D,
                                                    job_name + ".csv")
        line = np.array([
                            self.result.D,
                            self.result.N,
                            self.result.acc_mean,
                            self.result.acc_est_mean,
                            self.result.vol,
                            self.result.vol_est,
                            self.result.steps_taken
                         ])
        
        with open(fname, 'w+') as f:
            f.write(" ".join(map(str, line)))

def result_dict_from_file(fname):
    """
    Assumes a file with lots of lines as the one created by
    store_fire_and_forget_result and produces a dictionary with (D,N) as key
    and a Rx5 array with experimental results for each of the R repetitions
    """
    results = np.loadtxt(fname)
    
    result_dict = {}
    for i in range(len(results)):
        D = np.int(results[i, 0])
        N = np.int(results[i, 1])
        result_dict[(D, N)] = []

    for i in range(len(results)):
        D = np.int(results[i, 0])
        N = np.int(results[i, 1])
        acc_mean = results[i, 2]
        acc_est_mean = results[i, 3]
        vol = results[i, 4]
        vol_est = results[i, 5]
        steps_taken = results[i, 6]
        
        result_dict[(D, N)].append([acc_mean, acc_est_mean, vol, vol_est, steps_taken])
    
    for i in range(len(results)):
        D = np.int(results[i, 0])
        N = np.int(results[i, 1])
        result_dict[(D, N)] = np.asarray(result_dict[(D, N)])
    
    return result_dict
