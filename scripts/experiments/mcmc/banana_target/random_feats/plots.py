from prettytable import PrettyTable

from kmc.tools.latex_plot_init import plt
import numpy as np


def gen_sparse_2d_array_from_dict(dictionary, fun, default_value=np.nan):
    assert len(dictionary.keys()[0]) is 2
    
    Ns = np.sort(np.unique(np.array([N for (N, _) in dictionary.keys()])))
    Ds = np.sort(np.unique(np.array([D for (_, D) in dictionary.keys()])))
    
    a = np.zeros((len(Ns), len(Ds))) + default_value
    
    for (N, D), v in dictionary.items():
        a[Ns == N, Ds == D] = fun(v)
    
    return Ds, Ns, a

def gen_sparse_1d_array_from_dict(dictionary, fun, default_value=np.nan):
    assert np.isscalar(dictionary.keys()[0])
    
    Ds = np.sort(np.unique(np.array(dictionary.keys())))
    
    a = np.zeros(len(Ds)) + default_value
    
    for D, v in dictionary.items():
        a[Ds == D] = fun(v)
    
    return Ds, a

def print_table(values, title=""):
    t = PrettyTable("%s[%d]" % (title, i) for i in range(len(values)))
    t.add_row(values)
    if len(title) > 0:
        print(title)
    print(t)

def plot_banana_result_mean_N_D(results, D, stat_idx, normalise_by_time=False,
                                **kwargs):
    fun = lambda x:None
    Ds, Ns, _ = gen_sparse_2d_array_from_dict(results, fun)
    
    fun = lambda x: np.mean(x[:, 0])
    _, _, time_taken_set_up = gen_sparse_2d_array_from_dict(results, fun)
    
    fun = lambda x: np.mean(x[:, 1])
    _, _, time_taken_sampling = gen_sparse_2d_array_from_dict(results, fun)
    
    fun = lambda x: np.mean(x[:, stat_idx])
    _, _, avg = gen_sparse_2d_array_from_dict(results, fun)
    
    fun = lambda x: np.percentile(x[:, stat_idx], 25)
    _, _, lower = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, stat_idx], 75)
    _, _, upper = gen_sparse_2d_array_from_dict(results, fun)
    
    D_ind = Ds == D
    time_total = time_taken_sampling[:,D_ind]# + time_taken_set_up[:,D_ind]
    time_total = time_total.ravel()
    
    print_table(time_total, "Time total")
    
    normaliser = time_total if normalise_by_time else 1.0
    avg = avg[:, D_ind].ravel()/normaliser
    err = np.array([np.abs(avg-lower[:, D_ind].ravel()/normaliser),
                   np.abs(avg-upper[:, D_ind].ravel()/normaliser)])
    
    plt.plot(Ns, avg, kwargs['color'])
    plt.errorbar(Ns, avg, err, color=kwargs['color'])
    plt.grid(True)
    
    try:
        plt.title(kwargs['title'])
    except KeyError:
        pass
    
    try:
        plt.ylim(kwargs['ylim'])
    except KeyError:
        pass
    
    try:
        plt.xlim(kwargs['xlim'])
    except KeyError:
        pass
    
    try:
        plt.xlabel(kwargs['xlabel'])
    except KeyError:
        pass

def plot_banana_result_mean_D(results, D, stat_idx, normalise_by_time=False,
                                **kwargs):
    fun = lambda x:None
    Ds, _ = gen_sparse_1d_array_from_dict(results, fun)
    
    fun = lambda x: np.mean(x[:,0])
    _, time_taken_set_up = gen_sparse_1d_array_from_dict(results, fun)
    
    fun = lambda x: np.mean(x[:,1])
    _, time_taken_sampling = gen_sparse_1d_array_from_dict(results, fun)
    
    fun = lambda x: np.mean(x[:,stat_idx])
    _, avg = gen_sparse_1d_array_from_dict(results, fun)
    
    fun = lambda x: np.percentile(x[:, stat_idx], 25)
    _, lower = gen_sparse_1d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, stat_idx], 75)
    _, upper = gen_sparse_1d_array_from_dict(results, fun)
        
    D_ind = Ds == D
    time_total = time_taken_sampling[D_ind] + time_taken_set_up[D_ind]
    time_total = time_total.ravel()
    
    print_table(time_total, "Time total")
    
    normaliser = time_total if normalise_by_time else 1.0
    avg = avg[D_ind].ravel()/normaliser
    
    xlim=plt.xlim()
    plt.plot(xlim, [avg,avg], color=kwargs['color'])
    plt.plot(xlim, [lower,lower], '--', color=kwargs['color'])
    plt.plot(xlim, [upper,upper], '--', color=kwargs['color'])
    
    
    plt.grid(True)
    
    try:
        plt.title(kwargs['title'])
    except KeyError:
        pass
    
    try:
        plt.ylim(kwargs['ylim'])
    except KeyError:
        pass
    
    try:
        plt.xlabel(kwargs['xlabel'])
    except KeyError:
        pass
    

def plot_fixed_D(results, results_hmc, D):
    fun = lambda x: None
    _, _ = gen_sparse_1d_array_from_dict(results_hmc, fun)
    
    fun = lambda x: np.mean(x[:, 0])
    _, time_taken_set_up_hmc = gen_sparse_1d_array_from_dict(results_hmc, fun)
    
    fun = lambda x: np.mean(x[:, 1])
    _, time_taken_sampling_hmc = gen_sparse_1d_array_from_dict(results_hmc, fun)
    
    fun = lambda x: np.mean(x[:, 2])
    _, accept_hmc = gen_sparse_1d_array_from_dict(results_hmc, fun)
    
    fun = lambda x: np.mean(x[:, 3])
    _, avg_quantile_error_hmc = gen_sparse_1d_array_from_dict(results_hmc, fun)
    
    fun = lambda x: np.mean(x[:, 4])
    _, avg_ess_hmc = gen_sparse_1d_array_from_dict(results_hmc, fun)
    
    fun = lambda x: None
    Ds, Ns, _ = gen_sparse_2d_array_from_dict(results, fun)
    
    fun = lambda x: np.mean(x[:, 0])
    _, _, time_taken_set_up = gen_sparse_2d_array_from_dict(results, fun)
    
    fun = lambda x: np.mean(x[:, 1])
    _, _, time_taken_sampling = gen_sparse_2d_array_from_dict(results, fun)
    
    fun = lambda x: np.mean(x[:, 2])
    _, _, accept = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 2], 25)
    _, _, accept_lower25 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 2], 75)
    _, _, accept_upper75 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 2], 5)
    _, _, accept_lower5 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 2], 95)
    _, _, accept_upper95 = gen_sparse_2d_array_from_dict(results, fun)
    
    fun = lambda x: np.mean(x[:, 3])
    _, _, avg_quantile_error = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 3], 25)
    _, _, avg_quantile_error_lower25 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 3], 75)
    _, _, avg_quantile_error_upper75 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 3], 5)
    _, _, avg_quantile_error_lower5 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 3], 95)
    _, _, avg_quantile_error_upper95 = gen_sparse_2d_array_from_dict(results, fun)
    
    fun = lambda x: np.mean(x[:, 4])
    _, _, avg_ess = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 4], 25)
    _, _, avg_ess_lower25 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 4], 75)
    _, _, avg_ess_upper75 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 4], 5)
    _, _, avg_ess_lower5 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 4], 95)
    _, _, avg_ess_upper95 = gen_sparse_2d_array_from_dict(results, fun)
    
    fun = lambda x: np.mean(x[:, 5])
    _, _, avg_norm_of_mean = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 5], 25)
    _, _, avg_norm_of_mean_lower25 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 5], 75)
    _, _, avg_norm_of_mean_upper75 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 5], 5)
    _, _, avg_norm_of_mean_lower5 = gen_sparse_2d_array_from_dict(results, fun)
    fun = lambda x: np.percentile(x[:, 5], 95)
    _, _, avg_norm_of_mean_upper95 = gen_sparse_2d_array_from_dict(results, fun)
    
    D_ind = Ds == D
    time_total = time_taken_sampling[:,D_ind] + time_taken_set_up[:,D_ind]
    time_total_hmc = time_taken_sampling_hmc[D_ind] + time_taken_set_up_hmc[D_ind]
    
    
    plt.figure()
    plt.plot([Ns.min(), Ns.max()], [accept_hmc[D_ind], accept_hmc[D_ind]], 'r')
    plt.plot(Ns, accept[:, D_ind], 'b')
    plt.plot(Ns, accept_lower25[:, D_ind], 'b-.')
    plt.plot(Ns, accept_upper75[:, D_ind], 'b-.')
    plt.fill_between(Ns,
                     accept_lower5[:, D_ind].ravel(),
                     accept_upper95[:, D_ind].ravel(),
                     color="grey", alpha=.5)
    plt.title("Avg. acc. prob.")
    plt.ylim([0,1.1])
    plt.grid(True)
    plt.xlabel(r"$n$")
    
    plt.figure()
    plt.plot([Ns.min(), Ns.max()], [avg_quantile_error_hmc[D_ind], avg_quantile_error_hmc[D_ind]], 'r')
    plt.plot(Ns, avg_quantile_error[:, D_ind])
    plt.plot(Ns, avg_quantile_error_lower25[:, D_ind], 'b-.')
    plt.plot(Ns, avg_quantile_error_upper75[:, D_ind], 'b-.')
    plt.fill_between(Ns,
                     avg_quantile_error_lower5[:, D_ind].ravel(),
                     avg_quantile_error_upper95[:, D_ind].ravel(),
                     color="grey", alpha=.5)
    plt.title("Avg. quantile error")
    plt.grid(True)
    plt.xlabel(r"$n$")
    
    plt.figure()
#     plt.plot([Ns.min(), Ns.max()], np.array([avg_ess_hmc[D_ind], avg_ess_hmc[D_ind]])/time_total_hmc, 'r')
    plt.plot(Ns, avg_ess[:, D_ind]/time_total, 'b')
    plt.plot(Ns, avg_ess_lower25[:, D_ind]/time_total, 'b-.')
    plt.plot(Ns, avg_ess_upper75[:, D_ind]/time_total, 'b-.')
    
    plt.fill_between(Ns,
                     (avg_ess_lower5[:, D_ind]/time_total).ravel(),
                     (avg_ess_upper95[:, D_ind]/time_total).ravel(),
                     color="grey", alpha=.5)
    
    plt.title("Avg. ESS/s")
    plt.grid(True)
    plt.xlabel(r"$n$")
    
    
    plt.figure()
#     plt.plot([Ns.min(), Ns.max()], [avg_quantile_error_hmc[D_ind], avg_quantile_error_hmc[D_ind]], 'r')
    plt.plot(Ns, avg_norm_of_mean[:, D_ind])
    plt.plot(Ns, avg_norm_of_mean_lower25[:, D_ind], 'b-.')
    plt.plot(Ns, avg_norm_of_mean_upper75[:, D_ind], 'b-.')
    plt.fill_between(Ns,
                     avg_norm_of_mean_lower5[:, D_ind].ravel(),
                     avg_norm_of_mean_upper95[:, D_ind].ravel(),
                     color="grey", alpha=.5)
    plt.title(r"Avg. $\Vert \mathbb E \mathbf{x} \Vert$")
    plt.grid(True)
    plt.xlabel(r"$n$")
    
    
    plt.show()
    
