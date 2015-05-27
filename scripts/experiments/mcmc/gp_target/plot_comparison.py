from glob import glob
from matplotlib.lines import Line2D
import pickle

from kameleon_mcmc.kernel.PolynomialKernel import PolynomialKernel

from kmc.tools.Log import logger
from kmc.tools.latex_plot_init import plt
import numpy as np


if __name__ == "__main__":
    logger.setLevel(10)
    
    warmup = 200
    num_samples = 4800
    num_Ns = 10
    Ns = np.round(np.linspace(1, num_samples, num_Ns)).astype(int)
    num_repetitions = 3
    
    benchmark_samples = np.load("ground_truth/benchmark_samples.arr")

    fnames_kmc = glob('lite/KMC_N=1000_D=9_ground_truth_iterations=5200*.pkl')[:num_repetitions]
    fnames_kameleon = glob('kameleon/Kameleon_N=1000__ground_truth_iterations=5200*.pkl')[:num_repetitions]
    fnames_rw = glob('rw/RW_D=9_ground_truth_iterations*.pkl')[:num_repetitions]
    
    colors = ["b", "g", "m"]
    
    for alg_idx, fnames in enumerate([
                   fnames_kmc,
                   fnames_kameleon,
                   fnames_rw
                   ]):
        
        if len(fnames) <= 0:
            exit()
    
        MMDs = np.zeros((len(fnames), len(Ns)))
        k = PolynomialKernel(degree=3)
        
        for i, fname in enumerate(fnames):
            logger.info("%s" % fname)
            with open(fname) as f:
                result = pickle.load(f)
                
            # print some summary stats
            accepted = result.accepted
            avg_ess = result.posterior_statistics["avg_ess"]
            min_ess = result.posterior_statistics["min_ess"]
            
            time = result.time_taken_sampling
            time_set_up = result.time_taken_set_up
        
            logger.info("Average acceptance probability: %.2f" % np.mean(accepted))
            logger.info("Average ESS: %.2f" % avg_ess)
            logger.info("Minimum ESS: %.2f" % min_ess)
            total_time_s = time + time_set_up
            total_time_m = total_time_s / 60
            logger.info("Total time: %.2f sec, %.2f min" % (total_time_s, total_time_m))
            logger.info("Average ESS/s: %.2f" % (avg_ess / total_time_s))
            logger.info("Minimum ESS/s: %.2f" % (min_ess / total_time_s))
    #             plot_mcmc_result(result, D1=1, D2=6)
            
            # compute how MMD evolves
    
            for j, N in enumerate(Ns):
                logger.info("MMD of %d benchmark samples against %d MCMC samples" % (len(benchmark_samples), N))
                samples_so_far = result.samples[warmup:(warmup + N)]
                MMDs[i, j] = k.estimateMMD(benchmark_samples, samples_so_far)
        
        med =  np.median(MMDs, 0)
        lower =  np.percentile(MMDs, 20, 0)
        upper = np.percentile(MMDs, 80, 0)
    
        plt.plot(Ns, med, "-", color=colors[alg_idx])
        plt.plot(Ns, lower, '--', color=colors[alg_idx])
        plt.plot(Ns, upper, '--', color=colors[alg_idx])
#         err = np.array([np.abs(med-lower),np.abs(med-upper)])
#         plt.plot(Ns, med, color=colors[alg_idx])
#         plt.errorbar(Ns, med, err, color=colors[alg_idx])
        
    plt.yscale("log")
    line1 = Line2D([0, 0], [0, 0], color=colors[0])
    line2 = Line2D([0, 0], [0, 0], color=colors[1])
    line3 = Line2D([0, 0], [0, 0], color=colors[2])
    plt.legend((line1, line2, line3), ["KMC", "KAMH", "RW"])
    
    plt.ylabel(r"MMD from ground truth")
    plt.xlabel("Iterations")
    plt.grid(True)
    plt.savefig("gp_target_results.eps", bbox_inches='tight')
    plt.show()
