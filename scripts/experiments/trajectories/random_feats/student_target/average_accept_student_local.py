import os

from kmc.tools.Log import logger
import numpy as np
from scripts.experiments.trajectories.independent_jobs_classes.random_feats.StudentTrajectoryJob import StudentTrajectoryJob
from scripts.experiments.trajectories.tools import process


modulename = __file__.split(os.sep)[-1].split('.')[-2]

if __name__ == "__main__":
    logger.setLevel(10)
    nu_q = 1.
    sigma_p = 1.
    Ds = np.sort(2 ** np.arange(8))[::-1]
    Ns = np.sort([50, 100, 200, 500, 1000, 2000])[::-1]
    
    print(Ns)
    print(Ds)
    num_repetitions = 1
    num_steps = 100
    max_steps = 1000
    step_size = .1
    
    scale0 = 0.5
    lmbda0 = 0.00008
    
    job_generator = lambda D, N, m : StudentTrajectoryJob(N, D, m, nu_q,
                                                          sigma_p, num_steps,
                                                          step_size, scale0,
                                                          lmbda0, max_steps,
                                                          learn_parameters=True)
    
    process(modulename, job_generator, Ds, Ns, num_repetitions, num_steps,
            step_size, max_steps, compute_local=True)
