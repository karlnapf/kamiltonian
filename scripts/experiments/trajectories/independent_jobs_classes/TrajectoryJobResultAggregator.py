

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
                                                    job_name +  ".csv")
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

