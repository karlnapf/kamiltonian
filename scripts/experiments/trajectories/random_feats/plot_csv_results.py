import os

import matplotlib.pyplot as plt
from scripts.experiments.trajectories.independent_jobs_classes.TrajectoryJobResultAggregator import result_dict_from_file
from scripts.experiments.trajectories.plots import plot_trajectory_result_heatmap, \
    plot_trajectory_result_mean_fixed_D, plot_trajectory_result_mean_fixed_N, \
    gen_sparse_2d_array_from_dict, plot_trajectory_result_necessary_data


modulename = __file__.split(os.sep)[-1].split('.')[-2]

if __name__ == "__main__":
    fname = "results.csv"
    Ds, Ns, _ = gen_sparse_2d_array_from_dict(result_dict_from_file(fname), lambda x: None)

    plot_trajectory_result_heatmap(fname)
    
    for D in Ds:
        plot_trajectory_result_mean_fixed_D(fname, D=D)

    for N in Ns:
        plot_trajectory_result_mean_fixed_N(fname, N=N)
    
    plot_trajectory_result_necessary_data(fname, [0.1, 0.3, 0.5, 0.8])
    plt.show()
