from glob import glob
import pickle

from kmc.tools.Log import logger
import matplotlib.pyplot as plt
import numpy as np
from scripts.experiments.mcmc.independent_job_classes.debug import plot_mcmc_result


if __name__ == "__main__":
    logger.setLevel(10)
    
    thin = 300
    warmup = 1000
    
    samples = []
    
    fnames = glob('RW_D=10_ground_truth_iterations=3000_*.pkl')
    for fname in fnames:
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
            logger.info("Total time: %.2f" % (time + time_set_up))
            logger.info("Average ESS/s: %.2f" % (avg_ess / (time + time_set_up)))
            logger.info("Minimum ESS/s: %.2f" % (min_ess / (time + time_set_up)))
            
            if False:
                plot_mcmc_result(result, D1=1, D2=6)
        
        inds = np.arange(warmup, result.num_iterations, step=thin)
        for i in inds:
            samples += [result.samples[i]]
    
    samples = np.array(samples)
    samples = samples[np.random.permutation(len(samples))]
    logger.info("Extracted %d samples in dimension %d" % (samples.shape[0], samples.shape[1]))
    if True:
        with open("benchmark_samples.arr", 'w+') as f:
            np.save(f, samples)
    
    D_pairs = []
    
    for D1 in range(9):
        for D2 in range(D1):
            D_pairs += [(D1, D2)]
    
    for D1, D2 in D_pairs:
        print(D1, D2)
        plt.figure(figsize=(8, 12))
        plt.subplot(521)
        plt.plot(samples[:, D1])
        plt.subplot(522)
        plt.plot(samples[:, D2])
        plt.subplot(523)
        plt.hist(samples[:, D1])
        plt.subplot(524)
        plt.hist(samples[:, D2])
        plt.subplot(525)
        plt.plot(samples[:, D1], samples[:, D2])
        plt.subplot(526)
        plt.plot(samples[:, D1], samples[:, D2], '.')
        plt.show()
