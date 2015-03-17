import os

from kmc.scripts.experiments.trajectories.independent_jobs_classes.GaussianTrajectoryJob import GaussianTrajectoryJob
from kmc.scripts.experiments.trajectories.tools import process
import numpy as np


modulename = __file__.split(os.sep)[-1].split('.')[-2]

if __name__ == "__main__":
    sigma_q = 1.
    sigma_p = 1.
    Ds = 2 ** np.arange(11)
    num_repetitions = 100
    N = 500
    lmbda = 1.
    num_steps = 100
    max_steps = 1000
    step_size = .1
    
    job_generator = lambda D : GaussianTrajectoryJob(N, D, sigma_q, lmbda, sigma_p, num_steps, step_size, max_steps)
    
    process(modulename, job_generator, Ds, num_repetitions, N, lmbda, num_steps,
            step_size, max_steps)
