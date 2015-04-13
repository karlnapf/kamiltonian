import os

from kmc.tools.Log import logger
import matplotlib.pyplot as plt
import numpy as np
from scripts.experiments.trajectories.independent_jobs_classes.random_feats.GaussianTrajectoryJob import GaussianTrajectoryJob
from scripts.experiments.trajectories.plots import plot_trajectory_result_heatmap, \
    plot_trajectory_result_mean_median_fixed_N, \
    plot_trajectory_result_mean_median_fixed_D,\
    plot_trajectory_result_necessary_data
from scripts.experiments.trajectories.tools import process


modulename = __file__.split(os.sep)[-1].split('.')[-2]

if __name__ == "__main__":
    logger.setLevel(10)
    sigma_q = 1.
    sigma_p = 1.
    Ds = np.sort(2 ** np.arange(8))[::-1]
    Ns = np.sort([50, 100, 200, 500, 1000, 2000, 5000, 10000])[::-1]
    
    print(Ns)
    print(Ds)
    num_repetitions = 1
    num_steps = 100
    max_steps = 1000
    step_size = .1
    
    sigma0 = 0.5
    lmbda0 = 0.0001
    
    job_generator = lambda D, N, m : GaussianTrajectoryJob(N, D, m, sigma_q, sigma_p,
                                                     num_steps, step_size,
                                                     sigma0, lmbda0, max_steps)
    
    process(modulename, job_generator, Ds, Ns, num_repetitions, num_steps,
            step_size, max_steps, compute_local=True)


    fname = modulename + ".pkl"
    fname = fname.replace("_local", "")
    plot_trajectory_result_heatmap(fname)
      
    # slices of heatmap with confidence intervals
    for N in Ns:
        plot_trajectory_result_mean_median_fixed_N(fname, N=N)
          
    for D in Ds:
        plot_trajectory_result_mean_median_fixed_D(fname, D=D)
    
    plot_trajectory_result_necessary_data(fname, [0.1, 0.3, 0.5, 0.8])
