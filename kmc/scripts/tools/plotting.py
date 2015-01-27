from kameleon_mcmc.tools.Visualise import Visualise

from kmc.hamiltonian.leapfrog import leapfrog, compute_hamiltonian
import matplotlib.pyplot as plt
import numpy as np


def plot_array(Xs, Ys, D):
    """
    Plots a 2D array
    
    Xs - x values the density is evaluated at
    Ys - y values the density is evaluated at
    D - array to plot
    """
    im = plt.imshow(D, origin='lower')
    im.set_extent([Xs.min(), Xs.max(), Ys.min(), Ys.max()])
    im.set_interpolation('nearest')
    im.set_cmap('gray')
    plt.ylim([Ys.min(), Ys.max()])
    plt.xlim([Xs.min(), Xs.max()])
    
def evaluate_density_grid(Xs, Ys, log_pdf, exponentiate=False):
    D = np.zeros((len(Xs), len(Ys)))
    for i in range(len(Xs)):
        for j in range(len(Ys)):
            x = np.array([Xs[i], Ys[j]])
            D[j, i] = log_pdf(x)
    
    return D if exponentiate is False else np.exp(D)

def evaluate_density_grad_grid(Xs, Ys, grad):
    G = np.zeros((len(Xs), len(Ys)))
    for i in range(len(Xs)):
        for j in range(len(Ys)):
            x = np.array([Xs[i], Ys[j]])
            G[j, i] = np.linalg.norm(grad(x))
    
    return G

def plot_2d_trajectory(X):
    plt.plot(X[:, 0], X[:, 1], 'r-')
    plt.plot(X[0, 0], X[0, 1], 'r*', markersize=15)
    plt.plot(X[-1, 0], X[-1, 1], 'b*', markersize=15)

def plot_kamiltonian_dnyamics(q0, p0, logq, dlogq, logq_est, dlogq_est,
                              logp, dlogp, Z=None, num_steps=500, step_size=.1,
                              Xs_q=None, Ys_q=None, Xs_p=None, Ys_p=None,
                              plot_dlogq=False):
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
    Hs_est = compute_hamiltonian(Qs_est, Ps_est, logq_est, logp)
    
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
    
    plt.subplot(325)
    plt.title("Hamiltonian")
    plt.plot(Hs)
    
    plt.subplot(326)
    plt.title("Hamiltonian")
    plt.plot(Hs_est)
    
    plt.tight_layout()
    
