from kmc.tools.convergence_stats import autocorr
import matplotlib.pyplot as plt
import numpy as np

def plot_diagnosis(agg, D):
    plt.ioff()
    
    samples = agg.result.samples
    accepted = agg.result.accepted
    avg_quantile_errors = agg.result.posterior_statistics["avg_quantile_error"]
    avg_ess = agg.result.posterior_statistics["avg_ess"]
    norm_of_mean = agg.result.posterior_statistics["norm_of_mean"]
    
    time = agg.result.time_taken_sampling
    time_set_up = agg.result.time_taken_set_up
    
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
    print("Average quantile errors: %.2f" % avg_quantile_errors)
    
    print("Average acceptance probability: %.2f" % np.mean(accepted))
    print("Average ESS: %.2f" % avg_ess)
    print("Average ESS/s: %.2f" % (avg_ess / time))
    print("Average ESS/s including set up: %.2f" % (avg_ess / (time+time_set_up)))
    print("Average norm of mean: %.2f" % norm_of_mean)
    
    plt.show()

def plot_mcmc_result(result, D1, D2):
    plt.ioff()
    
    samples = result.samples
    accepted = result.accepted
    log_pdf = result.log_pdf
    avg_ess = result.posterior_statistics["avg_ess"]
    min_ess = result.posterior_statistics["min_ess"]
    
    time = result.time_taken_sampling
    time_set_up = result.time_taken_set_up
    
    # compute autocorrelation of the first dimension
    acorrs = autocorr(samples[:, D1])
    
    # visualise summary
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
    plt.plot(np.cumsum(accepted) / np.arange(1, len(samples) + 1))
    plt.subplot(526)
    plt.plot(acorrs)
    plt.subplot(527)
    plt.plot(samples[:,D1], samples[:,D2])
    plt.subplot(528)
    plt.plot(samples[:,D1], samples[:,D2], '.')
    plt.subplot(529)
    plt.plot(log_pdf, '-')
    
    # print quantile summary
    print("Average acceptance probability: %.2f" % np.mean(accepted))
    print("Average ESS: %.2f" % avg_ess)
    print("Minimum ESS: %.2f" % min_ess)
    print("Average ESS/s: %.2f" % (avg_ess / time))
    print("Minimum ESS/s: %.2f" % (min_ess / time))
    print("Average ESS/s including set up: %.2f" % (avg_ess / (time+time_set_up)))
    
    plt.show()
