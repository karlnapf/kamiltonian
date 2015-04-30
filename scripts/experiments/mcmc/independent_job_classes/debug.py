
from prettytable import PrettyTable

from kmc.tools.convergence_stats import autocorr
import matplotlib.pyplot as plt
import numpy as np

def plot_diagnosis(agg, D):
    plt.ioff()
    
    samples = agg.result.mcmc_job.samples
    accepted = agg.result.mcmc_job.accepted
    quantiles_emp = agg.result.mcmc_job.posterior_statistics["emp_quantiles"]
    ESS = agg.result.mcmc_job.posterior_statistics["ESS"]
    time = agg.result.mcmc_job.time_taken_sampling
    time_set_up = agg.result.mcmc_job.time_taken_set_up
    
    # compute autocorrelation of the dimension with heavier tails
    acorrs = autocorr(samples[:, 1])
    
    
    # visualise summary
    plt.figure(figsize=(8, 12))
    plt.subplot(421)
    plt.plot(samples[:, 0])
    plt.subplot(422)
    plt.plot(samples[:, 1])
    plt.subplot(423)
    plt.hist(samples[:, 0])
    plt.subplot(424)
    plt.hist(samples[:, 1])
    plt.subplot(425)
    plt.plot(np.cumsum(accepted) / np.arange(1, len(samples) + 1))
    plt.subplot(426)
    plt.plot(acorrs)
    plt.subplot(427)
    plt.plot(samples[:,0], samples[:,1])
    plt.subplot(428)
    plt.plot(samples[:,0], samples[:,1], '.')
    
    # print quantile summary
    print("Empirical quantiles")
    quantiles = np.arange(0.1, 1., 0.1)
    q_table = PrettyTable(["%.2f" % q for q in quantiles])
    q_table.add_row(quantiles_emp)
    print(q_table)
    
    print("Quantile errors")
    q_error_table = PrettyTable(["%.2f" % q for q in quantiles])
    q_error_table.add_row(np.abs(quantiles_emp - quantiles))
    print(q_error_table)
    avg_q_error = np.mean(np.abs(quantiles_emp - quantiles))
    print("Average quantile errors: %.2f" % avg_q_error)
    
    print("Average acceptance probability: %.2f" % np.mean(accepted))
    print("Average ESS: %.2f" % np.mean(ESS))
    print("Average ESS/s: %.2f" % (np.mean(ESS) / time))
    print("Average ESS/s including set up: %.2f" % (np.mean(ESS) / (time+time_set_up)))
    
    print("ESS for all dimensions:")
    ESS_table = PrettyTable(["%d" % d for d in range(D)])
    ESS_table.add_row(ESS)
    print(ESS_table)
    
    
    plt.show()
