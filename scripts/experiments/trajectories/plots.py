import matplotlib

from kmc.tools.Log import logger
from kmc.tools.latex_plot_init import plt
import numpy as np
from scripts.experiments.trajectories.independent_jobs_classes.TrajectoryJobResultAggregator import result_dict_from_file


def plot_trajectory_result_mean_fixed_N(fname, N):
    results = result_dict_from_file(fname)
    # acc_mean, acc_est_mean, vol, vol_est, steps_taken
    fun = lambda x: np.mean(x[:, 1])
    Ds, Ns, avg_accept_est_mean = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.mean(x[:, 0])
    _, _, avg_accept_mean = gen_sparse_2d_array_from_dict(results, fun)
    
    fun = lambda x: np.percentile(x[:, 1], 25)
    _, _, avg_accept_est_lower_25 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 1], 75)
    _, _, avg_accept_est_upper_25 = gen_sparse_2d_array_from_dict(results, fun)
    
    fun = lambda x: np.percentile(x[:, 1], 5)
    _, _, avg_accept_est_lower_5 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 1], 95)
    _, _, avg_accept_est_upper_95 = gen_sparse_2d_array_from_dict(results, fun)
    
    N_ind = np.where(Ns == N)[0][0]
    
    plt.figure()
    plt.plot(Ds, avg_accept_mean[:, N_ind], 'r')
    plt.plot(Ds, avg_accept_est_mean[:, N_ind], 'b')
    plt.plot(Ds, avg_accept_est_lower_25[:, N_ind], 'b-.')
    plt.plot(Ds, avg_accept_est_lower_5[:, N_ind], color="grey")
    plt.plot(Ds, avg_accept_est_upper_95[:, N_ind], color="grey")
    plt.fill_between(Ds, avg_accept_est_lower_5[:, N_ind],
                     avg_accept_est_upper_95[:, N_ind],
                     color="grey", alpha=.5)
    plt.plot(Ds, avg_accept_est_upper_25[:, N_ind], 'b-.')
    
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

def plot_trajectory_result_mean_fixed_D(fname, D):
    results = result_dict_from_file(fname)
    # acc_mean, acc_est_mean, vol, vol_est, steps_taken
    fun = lambda x: np.mean(x[:, 1])
    Ds, Ns, avg_accept_est_mean = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.mean(x[:, 0])
    _, _, avg_accept_mean = gen_sparse_2d_array_from_dict(results, fun)
    
    fun = lambda x: np.percentile(x[:, 1], 25)
    _, _, avg_accept_est_lower_25 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 1], 75)
    _, _, avg_accept_est_upper_25 = gen_sparse_2d_array_from_dict(results, fun)
    
    fun = lambda x: np.percentile(x[:, 1], 5)
    _, _, avg_accept_est_lower_5 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 1], 95)
    _, _, avg_accept_est_upper_95 = gen_sparse_2d_array_from_dict(results, fun)
    
    D_ind = np.where(Ds == D)[0][0]
    
    plt.figure()
    plt.plot(Ns, avg_accept_mean[D_ind, :], 'r')
    plt.plot(Ns, avg_accept_est_mean[D_ind, :], 'b')
    plt.plot(Ns, avg_accept_est_lower_25[D_ind, :], 'b-.')
    plt.plot(Ns, avg_accept_est_lower_5[D_ind, :], color="grey")
    plt.plot(Ns, avg_accept_est_upper_95[D_ind, :], color="grey")
    plt.fill_between(Ns, avg_accept_est_lower_5[D_ind, :],
                     avg_accept_est_upper_95[D_ind, :],
                     color="grey", alpha=.5)
    plt.plot(Ns, avg_accept_est_upper_25[D_ind, :], 'b-.')
    
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
    masked_array = np.ma.array (acc, mask=np.isnan(acc))
    cmap = matplotlib.cm.jet
    cmap.set_bad('w', 1.)
    
    plt.pcolor(Ns, Ds, masked_array, cmap=cmap)
    plt.yscale("log")
    plt.xlim([np.min(Ns), np.max(Ns)])
    plt.ylim([np.min(Ds), np.max(Ds)])
    plt.xlabel(r"$n$")
    plt.ylabel(r"$d$")
    plt.colorbar()

def gen_sparse_2d_array_from_dict(dictionary, fun, default_value=np.nan):
    assert len(dictionary.keys()[0]) is 2
    
    Ds = np.sort(np.unique(np.array([D for (D, _) in dictionary.keys()])))
    Ns = np.sort(np.unique(np.array([N for (_, N) in dictionary.keys()])))
    
    a = np.zeros((len(Ds), len(Ns))) + default_value
    for (D, N), v in dictionary.items():
        D_ind = np.where(Ds == D)[0][0]
        N_ind = np.where(Ns == N)[0][0]
        a[D_ind, N_ind] = fun(v)
    
    return Ds, Ns, a

def plot_trajectory_result_heatmap(fname):
    results = result_dict_from_file(fname)
    # acc_mean, acc_est_mean, vol, vol_est, steps_taken
    fun = lambda x: np.mean(x[:, 1])
    Ds, Ns, avg_accept_est = gen_sparse_2d_array_from_dict(results, fun)
    plt.figure()
    plot_acceptance_heatmap(Ns, Ds, avg_accept_est)
    plt.xscale('log')
    fname_base = fname.split(".")[-2]
    plt.savefig(fname_base + "_kmc.eps", axis_inches='tight')

def plot_trajectory_result_necessary_data(fname, accs_at_least=[0.5]):
    results = result_dict_from_file(fname)
    fun = lambda x: np.mean(x[:, 1])
    Ds, Ns, avg_accept_est = gen_sparse_2d_array_from_dict(results, fun)
    
    plt.figure()
    for acc_at_least in accs_at_least:
        N_at_least = np.zeros(len(Ds))
        for i, D in enumerate(Ds):
            w = np.where(avg_accept_est[i, :] > acc_at_least)[0]
            if len(w) > 0:
                N_at_least[i] = np.min(Ns[w])
                logger.info("%.2f acc. for D=%d at N=%d" % (acc_at_least, D, N_at_least[i]))
            else:
                logger.info("Did not reach %.2f acc. for D=%d" % (acc_at_least, D))
            
        plt.plot(Ds, N_at_least)
    plt.yscale('log')
#     plt.xscale('log')
    
    plt.legend(["%.2f acc." % acc_at_least for acc_at_least in accs_at_least], loc="lower right")
    plt.grid(True)
    
    fname_base = fname.split(".")[-2]
    plt.savefig(fname_base + "_data_needs_kmc.eps", axis_inches='tight')
