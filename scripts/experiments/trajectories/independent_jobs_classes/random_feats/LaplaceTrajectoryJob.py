from abc import abstractmethod

from kmc.tools.Log import logger
import numpy as np
from scripts.experiments.trajectories.independent_jobs_classes.random_feats.TrajectoryJob import TrajectoryJob


class LaplaceTrajectoryJob(TrajectoryJob):
    def __init__(self,
                 N, D, m,
                 scale_q, sigma_p,
                 num_steps, step_size, scale0=0.5, lmbda0=0.0001, max_steps=None):
        # note: using sigma0 for storing scale parameter
        TrajectoryJob.__init__(self, N, D, m, sigma_p,
                               num_steps, step_size, max_steps, scale0, lmbda0)
        
        self.scale_q = scale_q
    
    @abstractmethod
    def set_up(self):
        # target density, rough centred laplace distribution, isotropic
        self.dlogq = lambda x: np.array([(-1. if x[d] >= 0 else 1.) for d in range(len(x))]) / self.scale_q 
        self.logq = lambda x: -len(x)*np.log(2*self.scale_q)-np.sum(np.abs(x))/self.scale_q
    
        # starting state
        self.q_sample = lambda: np.random.laplace(loc=0., scale=self.scale_q, size=self.D)
        
        logger.info("N=%d, D=%d" % (self.N, self.D))
        self.Z = np.random.laplace(loc=0., scale=self.scale_q, size=(self.N, self.D))

    @abstractmethod
    def get_parameter_fname_suffix(self):
        suffix = "%s_scale=%.4f" % (self.__class__.__name__, self.scale_q)
        
        return suffix + "_" + TrajectoryJob.get_parameter_fname_suffix(self) 
