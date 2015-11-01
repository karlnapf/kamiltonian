import os

import matplotlib.pyplot as plt
from scripts.experiments.trajectories.independent_jobs_classes.TrajectoryJobResultAggregator import result_dict_from_file
from scripts.experiments.trajectories.plots import plot_trajectory_result_heatmap, \
    plot_trajectory_result_mean_fixed_D, plot_trajectory_result_mean_fixed_N, \
    gen_sparse_2d_array_from_dict, plot_trajectory_result_necessary_data,\
    plot_repetitions_heatmap


modulename = __file__.split(os.sep)[-1].split('.')[-2]

if __name__ == "__main__":
    fname = "rebuttal_target/average_accept_rebuttal_local.csv"
    fname = "rebuttal_target/average_accept_rebuttal.csv"
    fname = "gaussian_target/results.csv"
    
    
#     fname = "laplace_target/results.csv"
#     fname = "student_target/average_accept_student_local.csv"
    
    
    
    Ds, Ns, _ = gen_sparse_2d_array_from_dict(result_dict_from_file(fname), lambda x: None)

    plot_repetitions_heatmap(fname)
    plot_trajectory_result_heatmap(fname)
    plt.tight_layout()
    plt.savefig('average_accept_gaussian_target_kmc.eps')
    
    
    D = 16
    plot_trajectory_result_mean_fixed_D(fname, D=D)
    plt.tight_layout()
    plt.savefig('average_accept_gaussian_target_D=%d.eps' % D)
    
    N=2000
    plot_trajectory_result_mean_fixed_N(fname, N=N)
    plt.tight_layout()
    plt.savefig('average_accept_gaussian_target_N=%d.eps' % N)
    
      
    plot_trajectory_result_necessary_data(fname, [0.05, 0.2, 0.3, 0.5])
    plt.show()
