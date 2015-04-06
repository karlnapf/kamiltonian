import os

from kmc.tools.Log import logger
import numpy as np
from scripts.experiments.trajectories.independent_jobs_classes.random_feats.GaussianTrajectoryJob import GaussianTrajectoryJob
from scripts.experiments.trajectories.plots import plot_trajectory_result_mean_median_fixed_N,\
    plot_trajectory_result_mean_median_fixed_D
from scripts.experiments.trajectories.tools import process


modulename = __file__.split(os.sep)[-1].split('.')[-2]

if __name__ == "__main__":
    sigma_q = 1.
    sigma_p = 1.
    Ds = np.flipud(2 ** np.arange(8))
    Ns = np.flipud(np.array([50, 100, 200, 500, 1000, 2000]))
    print(Ns)
    print(Ds)
    num_repetitions = 30
    num_steps = 100
    max_steps = 1000
    step_size = .1
    
    sigma0 = 0.5
    lmbda0 = 0.0001
    
    job_generator = lambda D, N, m : GaussianTrajectoryJob(N, D, m, sigma_q, sigma_p,
                                                     num_steps, step_size,
                                                     sigma0, lmbda0, max_steps)
    
    process(modulename, job_generator, Ds, Ns, num_repetitions, num_steps,
            step_size, max_steps, compute_local=False)

    fname = modulename + ".npy"
    plot_trajectory_result_mean_median_fixed_N(fname, N=Ns[0])
    plot_trajectory_result_mean_median_fixed_D(fname, D=Ds[0])
    