from kmc.tools.latex_plot_init import plt
import numpy as np


def plot_trajectory_result_mean_median(fname):
    with open(fname, 'r') as f:
            results = np.load(f)
            avg_accept = results["avg_accept"]
            avg_accept_est = results["avg_accept_est"]
            vols = results["vols"]
            vols_est = results["vols_est"]
            Ds = results["Ds"]
    
    plt.plot(Ds, np.median(avg_accept, 0), 'r')
    plt.plot(Ds, np.percentile(avg_accept, 5, 0), 'r--')
    plt.plot(Ds, np.percentile(avg_accept, 95, 0), 'r--')
    
    plt.plot(Ds, np.median(avg_accept_est, 0), 'b')
    plt.plot(Ds, np.percentile(avg_accept_est, 5, 0), 'b--')
    plt.plot(Ds, np.percentile(avg_accept_est, 95, 0), 'b--')
    plt.plot(Ds, np.median(avg_accept_est, 0)-np.std(avg_accept_est, 0), 'b--')
    plt.xscale("log")
    plt.grid(True)
    plt.xlim([Ds.min(), Ds.max()])
    ylim = plt.ylim()
    plt.ylim([ylim[0], 1.01])
    
    plt.figure()
    plt.plot(Ds, np.median(vols, 0), 'r')
    plt.plot(Ds, np.percentile(vols, 5, 0), 'r.')
    plt.plot(Ds, np.percentile(vols, 95, 0), 'r.')
     
    plt.plot(Ds, np.median(vols_est, 0), 'b')
    plt.plot(Ds, np.percentile(vols_est, 5, 0), 'b.')
    plt.plot(Ds, np.percentile(vols_est, 95, 0), 'b.')
    plt.show()

def plot_trajectory_result_boxplot(fname):
    with open(fname, 'r') as f:
            results = np.load(f)
            avg_accept = results["avg_accept"]
            avg_accept_est = results["avg_accept_est"]
            Ds = results["Ds"]
    
    plt.figure()
    plt.boxplot(avg_accept_est)
    ylim = plt.ylim()
    plt.ylim([ylim[0], 1.01])
    plt.xticks(np.arange(1,len(Ds)+1), [str(D) for D in Ds])
    plt.grid(True)
    plt.figure()
    plt.boxplot(avg_accept)
    plt.ylim([ylim[0], 1.01])
    plt.xticks(np.arange(1,len(Ds)+1), [str(D) for D in Ds])
    plt.grid(True)
    plt.show()
    

def plot_trajectory_result_boxplot_mix(fname):
    with open(fname, 'r') as f:
            results = np.load(f)
            avg_accept = results["avg_accept"]
            avg_accept_est = results["avg_accept_est"]
            Ds = results["Ds"]
    
    plt.figure()
    plt.plot(np.log2(Ds)+1, np.mean(avg_accept, 0), 'ro')
    plt.boxplot(avg_accept_est)
    ylim = plt.ylim()
    plt.ylim([ylim[0], 1.01])
    plt.xticks(np.arange(1,len(Ds)+1), [str(D) for D in Ds])
    plt.grid(True)
#     plt.plot(np.log2(Ds)+1, np.percentile(avg_accept, 5, 0), 'r--')
    plt.legend(["HMC"], loc="lower left")
    plt.ylabel("Acceptance probability")
    plt.xlabel(r"$D$")
    fname_base = fname.split(".")[-2]
    plt.savefig(fname_base + ".eps", axis_inches='tight')
    plt.show()