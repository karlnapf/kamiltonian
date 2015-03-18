from abc import abstractmethod

from kmc.densities.banana import log_banana_pdf, sample_banana
from kmc.tools.Log import logger
from scripts.experiments.trajectories.independent_jobs_classes.lite.TrajectoryJob import TrajectoryJob


class BananaTrajectoryJob(TrajectoryJob):
    def __init__(self,
                 N, D, V, bananicity, lmbda,
                 sigma_p,
                 num_steps, step_size, max_steps=None):
        TrajectoryJob.__init__(self, N, D, lmbda, sigma_p, num_steps, step_size,
                               max_steps)
        
        self.V = V
        self.bananicity = bananicity

    @abstractmethod
    def set_up(self):
        # target density
        self.logq = lambda x: log_banana_pdf(x, self.bananicity, self.V)
        self.dlogq = lambda x: log_banana_pdf(x, self.bananicity, self.V, compute_grad=True)
        
        # starting state
        self.q_sample = lambda: sample_banana(1, self.D, self.bananicity, self.V)[0]
        
        logger.info("N=%d, D=%d" % (self.N, self.D))
        self.Z = sample_banana(self.N, self.D, self.bananicity, self.V)