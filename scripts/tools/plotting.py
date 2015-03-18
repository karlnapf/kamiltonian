from kmc.hamiltonian.hamiltonian import compute_hamiltonian,\
    compute_log_accept_pr, compute_log_det_trajectory
from kmc.hamiltonian.leapfrog import leapfrog
from kmc.tools.numerics import log_mean_exp
import matplotlib.pyplot as plt
import numpy as np


def plot_array(Xs, Ys, D, ax=None, plot_contour=False):
    """
    Plots a 2D array
    
    Xs - x values the density is evaluated at
    Ys - y values the density is evaluated at
    D - array to plot
    ax - optional axes object to plot on, default is gca()
    """
    
    if ax is None:
        ax = plt.gca()
    
    if plot_contour:
        im = ax.contour(Xs, Ys, D)
        plt.clabel(im, inline=1, fontsize=10)
        
    im = ax.imshow(D, origin='lower')
    im.set_extent([Xs.min(), Xs.max(), Ys.min(), Ys.max()])
    im.set_interpolation('nearest')
    im.set_cmap('gray')
    ax.set_ylim([Ys.min(), Ys.max()])
    ax.set_xlim([Xs.min(), Xs.max()])
    
def evaluate_density_grid(Xs, Ys, log_pdf, exponentiate=False):
    D = np.zeros((len(Xs), len(Ys)))
    for i in range(len(Xs)):
        for j in range(len(Ys)):
            x = np.array([Xs[i], Ys[j]])
            D[j, i] = log_pdf(x)
    
    return D if exponentiate is False else np.exp(D)

def evaluate_gradient_grid(Xs, Ys, grad_func):
    """
    Plot as
    quiver(X, Y, U, V, color='m')
    plot_array(Xs, Ys, G_norm)
    """
    G_norm=np.zeros((len(Ys), len(Xs)))
    X, Y = np.meshgrid(Xs, Ys)
    U = np.zeros(X.shape)
    V = np.zeros(Y.shape)
    for i in range(len(Xs)):
        for j in range(len(Ys)):
            x = np.array([Xs[i], Ys[j]])
            g = grad_func(x)
            G_norm[j,i] = np.linalg.norm(g)
            U[j, i] = g[0]
            V[j, i] = g[1]
    
    return G_norm, U, V, X, Y

def evaluate_density_grad_grid(Xs, Ys, grad):
    G = np.zeros((len(Xs), len(Ys)))
    for i in range(len(Xs)):
        for j in range(len(Ys)):
            x = np.array([Xs[i], Ys[j]])
            G[j, i] = np.linalg.norm(grad(x))
    
    return G

def plot_2d_trajectory(X):
    plt.plot(X[:, 0], X[:, 1], 'r-')
    plt.plot(X[0, 0], X[0, 1], 'r*', markersize=5)
    plt.plot(X[-1, 0], X[-1, 1], 'b*', markersize=5)

def plot_kamiltonian_dnyamics(q0, p0, logq, dlogq, logq_est, dlogq_est,
                              logp, dlogp, Z=None, num_steps=500, step_size=.1,
                              Xs_q=None, Ys_q=None, Xs_p=None, Ys_p=None,
                              plot_dlogq=False, plot_H_or_acc=True):
    D = len(q0)
    
    # compute and plot log-density, if D==2
    plt.figure(figsize=(12, 12))

    if D is 2:
        if Xs_q is None:
            Xs_q = np.linspace(-3, 3)
        
        if Ys_q is None:
            Ys_q = np.linspace(-3, 3)
        
        if Xs_p is None:
            Xs_p = np.linspace(-3, 3)
        
        if Ys_p is None:
            Ys_p = np.linspace(-3, 3)
        
        if plot_dlogq:
            G = evaluate_density_grad_grid(Xs_q, Ys_q, dlogq)
            G_est = evaluate_density_grad_grid(Xs_q, Ys_q, dlogq_est)
        else:
            G = evaluate_density_grid(Xs_q, Ys_q, logq)
            G_est = evaluate_density_grid(Xs_q, Ys_q, logq_est)
            
        M = evaluate_density_grid(Xs_p, Ys_p, logp)
        M_est = evaluate_density_grid(Xs_p, Ys_p, logp)
    
        plt.subplot(321)
        if not plot_dlogq:
            plot_array(Xs_q, Ys_q, np.exp(G))
        else:
            plot_array(Xs_q, Ys_q, G)
            
        if Z is not None:
            plt.plot(Z[:, 0], Z[:, 1], 'bx')
            
        plt.subplot(322)
        plot_array(Xs_q, Ys_q, np.exp(G_est))
        if Z is not None:
            plt.plot(Z[:, 0], Z[:, 1], 'bx')
        
        plt.subplot(323)
        plot_array(Xs_p, Ys_p, np.exp(M))
        plt.subplot(324)
        plot_array(Xs_p, Ys_p, np.exp(M_est))
    
    Qs, Ps = leapfrog(q0, dlogq, p0, dlogp, step_size, num_steps)
    Qs_est, Ps_est = leapfrog(q0, dlogq_est, p0, dlogp, step_size, num_steps)
    Hs = compute_hamiltonian(Qs, Ps, logq, logp)
    Hs_est = compute_hamiltonian(Qs_est, Ps_est, logq, logp)
    
    log_acc = compute_log_accept_pr(q0, p0, Qs, Ps, logq, logp)
    log_acc_est = compute_log_accept_pr(q0, p0, Qs_est, Ps_est, logq, logp)
    acc_mean = np.exp(log_mean_exp(log_acc))
    acc_est_mean = np.exp(log_mean_exp(log_acc_est))
    print "HMC acceptance prob: %.2f" % acc_mean
    print "KMC acceptance prob: %.2f" % acc_est_mean

    spread = compute_log_det_trajectory(Qs, Ps)
    spread_est = compute_log_det_trajectory(Qs_est, Ps_est)
    print "HMC spread: %.2f" % (spread)
    print "KMC spread: %.2f" % (spread_est)
    
    plt.subplot(321)
    plot_2d_trajectory(Qs)
    plt.title("True density")
    plt.subplot(322)
    plot_2d_trajectory(Qs_est)
    plt.title("Estimated density")
    
    plt.subplot(323)
    plt.title("Momentum")
    plot_2d_trajectory(Ps)
    plt.subplot(324)
    plt.title("Momentum")
    plot_2d_trajectory(Ps_est)
    
    if plot_H_or_acc:
        ylim = [np.min([Hs.min(), Hs_est.min()]),
                np.max([Hs.max(), Hs_est.max()])]
            
        plt.subplot(325)
        plt.title("Hamiltonian")
        plt.plot(Hs)
        plt.ylim(ylim)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        
        plt.subplot(326)
        plt.title("Hamiltonian")
        plt.plot(Hs_est)
        plt.ylim(ylim)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    else:
        plt.subplot(325)
        plt.title("Acceptance prob.")
        plt.plot(np.exp(log_acc))
        plt.plot([0,len(log_acc)], [acc_mean, acc_mean], "r")
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        
        plt.subplot(326)
        plt.title("Acceptance prob.")
        plt.plot(np.exp(log_acc_est))
        plt.plot([0,len(log_acc_est)], [acc_est_mean, acc_est_mean], "r")
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        
    plt.tight_layout()
    
