from kmc.tools.Log import logger
from kmc.tools.latex_plot_init import plt
import numpy as np


def plot_trajectory_result_mean_median_fixed_N(fname, N):
    with open(fname, 'r') as f:
        # created as: avg_accept = np.zeros((num_repetitions, len(Ds), len(Ns)))
        results = np.load(f)
        Ns = results['Ns']
        if not N in Ns:
            raise ValueError("Provided N (%d) is not in experiment" % N)
        
        N_idx = np.where(Ns == N)[0][0]
        Ds = results["Ds"]
        avg_accept = results["avg_accept"][:, :, N_idx]
        avg_accept_est = results["avg_accept_est"][:, :, N_idx]
        vols = results["vols"][:, :, N_idx]
        vols_est = results["vols_est"][:, :, N_idx]
    
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
    
    plt.xscale("log")
    plt.grid(True)
    plt.xlim([Ds.min(), Ds.max()])
    plt.xlabel(r"$d$")
    plt.title(r"n=%d" % N)
    ylim = plt.ylim()
    plt.ylim([ylim[0], 1.01])
    
    plt.legend(["HMC", "KMC median", "KMC 25\%-75\%", "KMC 5\%-95\%"], loc="lower left")
    fname_base = fname.split(".")[-2]
    plt.savefig(fname_base + "_N=%d.eps" % N, axis_inches='tight')
    
def plot_trajectory_result_mean_median_fixed_D(fname, D):
    with open(fname, 'r') as f:
        # created as: avg_accept = np.zeros((num_repetitions, len(Ds), len(Ns)))
        results = np.load(f)
        Ns = results['Ns']
        Ds = results["Ds"]
        if not D in Ds:
            raise ValueError("Provided D (%d) is not in experiment" % D)
        
        D_idx = np.where(Ds == D)[0][0]
        avg_accept = results["avg_accept"][:, D_idx, :]
        avg_accept_est = results["avg_accept_est"][:, D_idx, :]
        vols = results["vols"][:, D_idx, :]
        vols_est = results["vols_est"][:, D_idx, :]
    
    plt.figure()
    plt.plot(Ns, np.median(avg_accept, 0), 'r')
    plt.plot(Ns, np.median(avg_accept_est, 0), 'b')
    plt.plot(Ns, np.percentile(avg_accept_est, 25, 0), 'b-.')
    plt.plot(Ns, np.percentile(avg_accept_est, 5, 0), color="grey")
    plt.plot(Ns, np.percentile(avg_accept_est, 95, 0), color="grey")
    plt.fill_between(Ns, np.percentile(avg_accept_est, 5, 0),
                     np.percentile(avg_accept_est, 95, 0),
                     color="grey", alpha=.5)
    plt.plot(Ns, np.percentile(avg_accept_est, 75, 0), 'b-.')
    plt.plot(Ns, np.median(avg_accept, 0), 'r')
    
    plt.xscale("log")
    plt.grid(True)
    plt.xlim([Ns.min(), Ns.max()])
    plt.xlabel(r"$n$")
    ylim = plt.ylim()
    plt.ylim([ylim[0], 1.01])
    plt.title(r"d=%d" % D)
    
#     plt.legend(["HMC", "KMC median", "KMC 25\%-75\%", "KMC 5\%-95\%"], loc="lower left")
    fname_base = fname.split(".")[-2]
    plt.savefig(fname_base + "_D=%d.eps" % D, axis_inches='tight')

def plot_acceptance_heatmap(Ns, Ds, acc):
    plt.pcolor(Ns, Ds, acc)
    plt.yscale("log")
    plt.xlim([np.min(Ns), np.max(Ns)])
    plt.ylim([np.min(Ds), np.max(Ds)])
    plt.xlabel(r"$n$")
    plt.ylabel(r"$d$")
    plt.colorbar()

def plot_trajectory_result_heatmap(fname):
    with open(fname, 'r') as f:
        # created as: avg_accept = np.zeros((num_repetitions, len(Ds), len(Ns)))
        results = np.load(f)
        Ns = results['Ns']
        Ds = results["Ds"]
        avg_accept_est = np.mean(results["avg_accept_est"], 0)
    
    plt.figure()
    plot_acceptance_heatmap(Ns, Ds, avg_accept_est)
    plt.xscale('log')
    fname_base = fname.split(".")[-2]
    plt.savefig(fname_base + "_kmc.eps", axis_inches='tight')

def plot_trajectory_result_necessary_data(fname, accs_at_least=[0.5]):
    with open(fname, 'r') as f:
        # created as: avg_accept = np.zeros((num_repetitions, len(Ds), len(Ns)))
        results = np.load(f)
        Ns = results['Ns']
        Ds = results["Ds"]
        avg_accept_est = np.mean(results["avg_accept_est"], 0)
    
    
    plt.figure()
    for acc_at_least in accs_at_least:
        N_at_least = np.zeros(len(Ds))
        for i, D in enumerate(Ds):
            w = np.where(avg_accept_est[i,:]>acc_at_least)[0]
            if len(w)>0:
                N_at_least[i] = np.min(Ns[w])
                logger.info("%.2f acc. for D=%d at N=%d" % (acc_at_least, D, N_at_least[i]))
            else:
                logger.info("Did not reach %.2f acc. for D=%d" % (acc_at_least, D))
            
        plt.plot(Ds, N_at_least)
    plt.yscale('log')
    plt.xscale('log')
    
    plt.legend(["%.2f acc." % acc_at_least for acc_at_least in accs_at_least], loc="lower right")
    plt.grid(True)
    
    fname_base = fname.split(".")[-2]
    plt.savefig(fname_base + "_data_needs_kmc.eps", axis_inches='tight')