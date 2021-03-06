from matplotlib.lines import Line2D
import os

import matplotlib.pyplot as plt
from scripts.experiments.mcmc.banana_target.random_feats.plots import plot_banana_result_mean_N_D,\
    plot_banana_result_mean_D
from scripts.experiments.mcmc.independent_job_classes.KMCRandomFeatsJob import KMCRandomFeatsJob
from scripts.experiments.mcmc.independent_job_classes.MCMCJob import MCMCJob


modulename = __file__.split(os.sep)[-1].split('.')[-2]

if __name__ == "__main__":
    fname_hmc = "results_hmc.csv"
    fname_kmc = "results_kmc.csv"
    fname_rw = "results_rw.csv"
    fname_kameleon = "results_kameleon.csv"
    
    resuts_hmc = MCMCJob.result_dict_from_file(fname_hmc)
    resuts_kmc = KMCRandomFeatsJob.result_dict_from_file(fname_kmc)
    resuts_rw = MCMCJob.result_dict_from_file(fname_rw)
    resuts_kameleon = KMCRandomFeatsJob.result_dict_from_file(fname_kameleon)
    
    for D in [8]:
        xmax = 2000
        normalise_by_time = False
        titles = {
            0: 'Time set up',
            1: 'Time sampling',
            2: 'Acc. rate',
            3: 'Minimum ESS',
                  5: r'$\Vert \hat{\mathbb E} [X]\Vert$',
                  4: r'Quantile Error',
                  6: 'Average ESS',
                  }
        
        for stat_idx in [2,3,6,5]:
            
            title = '%s' % (titles[stat_idx])
            xlabel=r'$n$'
            plot_banana_result_mean_N_D(resuts_kameleon, D, stat_idx=stat_idx, normalise_by_time=normalise_by_time, title=title, xlabel=xlabel, color='g', xlim=[0, xmax])
            plot_banana_result_mean_D(resuts_hmc, D, stat_idx=stat_idx, normalise_by_time=normalise_by_time, plot_error=False, title=title, xlabel=xlabel, color='r')
            plot_banana_result_mean_N_D(resuts_kmc, D, stat_idx=stat_idx, normalise_by_time=normalise_by_time, title=title, xlabel=xlabel, color='b', xlim=[0, xmax])
            plot_banana_result_mean_D(resuts_rw, D, stat_idx=stat_idx, normalise_by_time=normalise_by_time, plot_error=False, title=title, xlabel=xlabel, color='m')
            
            line1 = Line2D([0,0], [0,0], color='r')
            line2 = Line2D([0,0], [0,0], color='b')
            line3 = Line2D([0,0], [0,0], color='m')
            line4 = Line2D([0,0], [0,0], color='g')
            
            if stat_idx ==3:
                plt.legend( (line1, line2, line3, line4), ('HMC', 'KMC', 'RW', 'KAMH'), loc='upper left')
            
            plt.xlabel("$n$")
            plt.tight_layout()
            plt.show()
