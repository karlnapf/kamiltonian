import os

import matplotlib.pyplot as plt
from scripts.experiments.mcmc.independent_job_classes.HMCJob import HMCJob
from scripts.experiments.mcmc.independent_job_classes.KMCRandomFeatsJob import KMCRandomFeatsJob
from scripts.experiments.mcmc.independent_job_classes.MCMCJob import MCMCJob
from scripts.experiments.mcmc.plots import plot_fixed_D, \
    plot_banana_result_mean_N_D, plot_banana_result_mean_D


modulename = __file__.split(os.sep)[-1].split('.')[-2]

if __name__ == "__main__":
    fname_hmc = "random_feats/banana_target/results_hmc.csv"
    fname_kmc = "random_feats/banana_target/results_kmc.csv"
    fname_rw = "random_feats/banana_target/results_rw.csv"
    fname_kameleon = "random_feats/banana_target/results_kameleon.csv"
    
    resuts_hmc = MCMCJob.result_dict_from_file(fname_hmc)
    resuts_kmc = KMCRandomFeatsJob.result_dict_from_file(fname_kmc)
    resuts_rw = MCMCJob.result_dict_from_file(fname_rw)
    resuts_kameleon = KMCRandomFeatsJob.result_dict_from_file(fname_kameleon)

    for D in [2, 8, 16]:
        normalise_by_time = False
        xmax = 2000
        titles = {
            0: 'Time set up',
            1: 'Time sampling',
            2: 'Acceptance',
            3: 'Quantile error',
                  4: r'$\Vert \mathbb E [X]\Vert$',
                  5: 'ESS',
                  
                  }
        
        
        for stat_idx in range(len(titles)):
            title = '%s, D=%d' % (titles[stat_idx], D)
            plot_banana_result_mean_N_D(resuts_kmc, D, stat_idx=stat_idx, normalise_by_time=normalise_by_time, title=title, xlabel='', color='b', xlim=[0, xmax])
            plot_banana_result_mean_D(resuts_hmc, D, stat_idx=stat_idx, normalise_by_time=normalise_by_time, plot_error=False, title=title, xlabel='', color='r')
            plot_banana_result_mean_N_D(resuts_kameleon, D, stat_idx=stat_idx, normalise_by_time=normalise_by_time, title=title, xlabel='', color='g', xlim=[0, xmax])
            plot_banana_result_mean_D(resuts_rw, D, stat_idx=stat_idx, normalise_by_time=normalise_by_time, plot_error=False, title=title, xlabel='', color='m')
            
    #         plt.legend(["KMC", "HMC", "MH", "KMH"])
            plt.show()
