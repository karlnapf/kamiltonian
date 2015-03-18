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
    
    plt.figure()
    plt.plot(Ds, np.median(avg_accept, 0), 'r')
    plt.plot(Ds, np.median(avg_accept_est, 0), 'b')
    plt.plot(Ds, np.percentile(avg_accept_est, 25, 0), 'b-.')
    plt.plot(Ds, np.percentile(avg_accept_est, 5, 0), color="grey")
    plt.plot(Ds, np.percentile(avg_accept_est, 95, 0), color="grey")
    plt.fill_between(Ds, np.percentile(avg_accept_est, 5, 0),
                     np.percentile(avg_accept_est, 95, 0),
                     color="grey", alpha=.5)
    plt.plot(Ds, np.percentile(avg_accept_est, 75, 0), 'b-.')
    plt.plot(Ds, np.median(avg_accept, 0), 'r')
    
#     plt.plot(Ds, np.percentile(avg_accept, 5, 0), 'r--')
#     plt.plot(Ds, np.percentile(avg_accept, 95, 0), 'r--')
    
    plt.xscale("log")
    plt.grid(True)
    plt.xlim([Ds.min(), Ds.max()])
    ylim = plt.ylim()
    plt.ylim([ylim[0], 1.01])
    
    plt.legend(["HMC", "KMC median", "KMC 25\%-75\%", "KMC 5\%-95\%"], loc="lower left")
    fname_base = fname.split(".")[-2]
    plt.savefig(fname_base + ".eps", axis_inches='tight')
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
    plt.xticks(np.arange(1, len(Ds) + 1), [str(D) for D in Ds])
    plt.grid(True)
    plt.figure()
    plt.boxplot(avg_accept)
    plt.ylim([ylim[0], 1.01])
    plt.xticks(np.arange(1, len(Ds) + 1), [str(D) for D in Ds])
    plt.grid(True)
    fname_base = fname.split(".")[-2]
    plt.savefig(fname_base + ".eps", axis_inches='tight')
    plt.show()
    

def plot_trajectory_result_boxplot_mix(fname):
    with open(fname, 'r') as f:
            results = np.load(f)
            avg_accept = results["avg_accept"]
            avg_accept_est = results["avg_accept_est"]
            Ds = results["Ds"]
    
    plt.figure()
    plt.plot(np.log2(Ds) + 1, np.mean(avg_accept, 0), 'ro')
    plt.boxplot(avg_accept_est)
    ylim = plt.ylim()
    plt.ylim([ylim[0], 1.01])
    plt.xticks(np.arange(1, len(Ds) + 1), [r"$2^{"+str(np.int(np.log2(D))) + "}$" for D in Ds])
    plt.gca().yaxis.grid(True)
    plt.plot(np.log2(Ds) + 1, np.mean(avg_accept, 0), 'ro')
#     plt.plot(np.log2(Ds)+1, np.percentile(avg_accept, 5, 0), 'r--')
    plt.legend(["HMC"], loc="lower left")
    plt.ylabel("Acceptance probability")
    plt.xlabel(r"$D$")
    fname_base = fname.split(".")[-2]
    plt.savefig(fname_base + ".eps", axis_inches='tight')
    plt.show()
