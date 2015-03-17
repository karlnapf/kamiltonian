import os

from kmc.scripts.experiments.trajectories.independent_jobs_classes.random_feats.GaussianTrajectoryJob import GaussianTrajectoryJob
from kmc.scripts.experiments.trajectories.tools import process
import numpy as np


modulename = __file__.split(os.sep)[-1].split('.')[-2]

if __name__ == "__main__":
    sigma_q = 1.
    sigma_p = 1.
    Ds = 2 ** np.arange(5)
    num_repetitions = 100
    N = 100
    lmbda = 0.0001
    m = N
    num_steps = 100
    max_steps = 1000
    step_size = .1
    
    sigma0 = 0.5
    lmbda0 = 0.0001
    
    job_generator = lambda D : GaussianTrajectoryJob(N, D, m, sigma_q, sigma_p,
                                                     num_steps, step_size,
                                                     sigma0, lmbda0, max_steps)
    
    process(modulename, job_generator, Ds, num_repetitions, N, lmbda, num_steps,
            step_size, max_steps)
