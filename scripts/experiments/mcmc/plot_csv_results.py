import os

import matplotlib.pyplot as plt
from scripts.experiments.mcmc.independent_job_classes.HMCJob import HMCJob
from scripts.experiments.mcmc.independent_job_classes.KMCRandomFeatsJob import KMCRandomFeatsJob
from scripts.experiments.mcmc.plots import plot_fixed_D,\
    plot_banana_result_mean_N_D, plot_banana_result_mean_D


modulename = __file__.split(os.sep)[-1].split('.')[-2]

if __name__ == "__main__":
    fname_hmc = "random_feats/banana_target/results_hmc.csv"
    fname_kmc = "random_feats/banana_target/results_kmc.csv"
    fname_rw = "random_feats/banana_target/results_rw.csv"
    fname_kameleon = "random_feats/banana_target/results_kmh.csv"
    
    resuts_hmc = HMCJob.result_dict_from_file(fname_hmc)
    resuts_kmc = KMCRandomFeatsJob.result_dict_from_file(fname_kmc)
    resuts_rw = KMCRandomFeatsJob.result_dict_from_file(fname_rw)
    resuts_kameleon = KMCRandomFeatsJob.result_dict_from_file(fname_kameleon)

    for D in [2, 8, 16, 24, 32]:
        if D is 2:
            normalise_by_time = True
        else:
            normalise_by_time = False
        
        plot_banana_result_mean_N_D(resuts_kmc, D, stat_idx=4, normalise_by_time=normalise_by_time, title='Avg. ESS/s, D=%d' % D, xlabel='', color='b', xlim=[0,2000])
        plot_banana_result_mean_D(resuts_hmc, D, stat_idx=4, normalise_by_time=normalise_by_time, plot_error=False, title='Avg. ESS/s, D=%d' % D, xlabel='', color='r')
        
#         plt.legend(["KMC", "HMC", "MH", "KMH"])
        plt.show()