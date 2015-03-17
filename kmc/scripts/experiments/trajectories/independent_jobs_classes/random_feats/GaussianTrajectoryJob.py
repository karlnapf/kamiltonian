from abc import abstractmethod

from kmc.densities.gaussian import log_gaussian_pdf, sample_gaussian
from kmc.scripts.experiments.trajectories.independent_jobs_classes.random_feats.TrajectoryJob import TrajectoryJob
from kmc.tools.Log import logger
import numpy as np


class GaussianTrajectoryJob(TrajectoryJob):
    def __init__(self,
                 N, D, m,
                 sigma_q, sigma_p,
                 num_steps, step_size, sigma0=0.5, lmbda0=0.0001, max_steps=None):
        TrajectoryJob.__init__(self, N, D, m, sigma_p,
                               num_steps, step_size, max_steps, sigma0, lmbda0)
        
        self.sigma_q = sigma_q
    
    @abstractmethod
    def set_up(self):
        L = np.linalg.cholesky(np.eye(self.D) * self.sigma_q)
        
        # target density
        self.dlogq = lambda x: log_gaussian_pdf(x, Sigma=L, is_cholesky=True, compute_grad=True)
        self.logq = lambda x: log_gaussian_pdf(x, Sigma=L, is_cholesky=True, compute_grad=False)
    
        # starting state
        self.q_sample = lambda: sample_gaussian(N=1, mu=np.zeros(self.D), Sigma=L, is_cholesky=True)[0]
        
        logger.info("N=%d, D=%d" % (self.N, self.D))
        self.Z = sample_gaussian(self.N, mu=np.zeros(self.D), Sigma=L, is_cholesky=True)
