from abc import abstractmethod

from kmc.densities.student import log_student_pdf
from kmc.tools.Log import logger
import numpy as np
from scripts.experiments.trajectories.independent_jobs_classes.random_feats.TrajectoryJob import TrajectoryJob


class StudentTrajectoryJob(TrajectoryJob):
    def __init__(self,
                 N, D, m,
                 nu_q, sigma_p,
                 num_steps, step_size, sigma0=0.5, lmbda0=0.0001, max_steps=None, learn_parameters=False):
        # note: using sigma0 for storing scale parameter
        TrajectoryJob.__init__(self, N, D, m, sigma_p,
                               num_steps, step_size, max_steps, sigma0, lmbda0, learn_parameters)
        
        self.nu_q = nu_q
    
    @abstractmethod
    def set_up(self):
        # target density, rough centred laplace distribution, isotropic
        self.dlogq = lambda x: log_student_pdf(x, self.nu_q, True)
        self.logq = lambda x: log_student_pdf(x, self.nu_q, False)
    
        # starting state
        self.q_sample = lambda: np.random.standard_t(df=self.nu_q, size=self.D)
        
        logger.info("N=%d, D=%d" % (self.N, self.D))
        self.Z = np.random.standard_t(df=self.nu_q, size=(self.N, self.D))

    @abstractmethod
    def get_parameter_fname_suffix(self):
        suffix = "%s_nu=%.4f" % (self.__class__.__name__, self.nu_q)
        
        return suffix + "_" + TrajectoryJob.get_parameter_fname_suffix(self) 
