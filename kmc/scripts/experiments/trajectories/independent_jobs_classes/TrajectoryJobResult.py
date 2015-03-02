
from independent_jobs.results.JobResult import JobResult


class TrajectoryJobResult(JobResult):
    def __init__(self, acc_mean, acc_est_mean, vol, vol_est):
        JobResult.__init__(self)
        
        self.acc_mean = acc_mean
        self.acc_est_mean = acc_est_mean
        self.vol = vol
        self.vol_est = vol_est
