import os

from kmc.densities.banana import log_banana_pdf, sample_banana
from kmc.densities.gaussian import log_gaussian_pdf, sample_gaussian
from kmc.hamiltonian.leapfrog import leapfrog, compute_hamiltonian
from kmc.score_matching.estimator import log_pdf_estimate, log_pdf_estimate_grad
from kmc.score_matching.gaussian_rkhs import _compute_b_sym, _compute_C_sym, \
    score_matching_sym, _objective_sym, xvalidate
from kmc.score_matching.gaussian_rkhs_xvalidation import select_sigma_grid
from kmc.score_matching.kernel.kernels import gaussian_kernel, \
    gaussian_kernel_grad
from kmc.scripts.tools.plotting import evaluate_density_grid, plot_array, \
    plot_2d_trajectory
from kmc.tools.latex_plot_init import plt
import numpy as np


fname_base = __file__.split(os.sep)[-1].split(".")[-2]

if __name__ == "__main__":
    s = 12
    np.random.seed(s)
    
    D = 2
    
    # true target log density
    logq = lambda x: log_banana_pdf(x, compute_grad=False)
    dlogq = lambda x: log_banana_pdf(x, compute_grad=True)

    # estimate density in rkhs
    N = 200
    mu = np.zeros(D)
    Z = sample_banana(N, D)
    lmbda = 1.
    sigma = select_sigma_grid(Z, lmbda=lmbda)
    
    K = gaussian_kernel(Z, sigma=sigma)
    b = _compute_b_sym(Z, K, sigma)
    C = _compute_C_sym(Z, K, sigma)
    a = score_matching_sym(Z, sigma, lmbda, K, b, C)
    J = _objective_sym(Z, sigma, lmbda, a, K, b, C)
    J_xval = np.mean(xvalidate(Z, 5, sigma, lmbda, K))
    print "N=%d, sigma: %.2f, lambda: %.2f, J(a)=%.2f, XJ(a)=%.2f" % \
            (N, sigma, lmbda, J, J_xval)
    
    kernel = lambda X, Y = None: gaussian_kernel(X, Y, sigma=sigma)
    kernel_grad = lambda x, X = None: gaussian_kernel_grad(x, X, sigma)
    logq_est = lambda x: log_pdf_estimate(x, a, Z, kernel)
    dlogq_est = lambda x: log_pdf_estimate_grad(x, a, Z, kernel_grad)
    
    # momentum
    Sigma_p = np.eye(D) * .1
    L_p = np.linalg.cholesky(Sigma_p)
    logp = lambda x: log_gaussian_pdf(x, Sigma=L_p, compute_grad=False, is_cholesky=True)
    dlogp = lambda x: log_gaussian_pdf(x, Sigma=L_p, compute_grad=True, is_cholesky=True)
    p_sample = lambda: sample_gaussian(N=1, mu=np.zeros(D), Sigma=L_p, is_cholesky=True)[0]

    # starting state
    p0 = p_sample()
    q0 = np.zeros(D)
    q0[:2] = np.array([0, -3])
    print "Starting at"
    print "p0", p0
    print "q0", q0
    
    # parameters
    num_steps = 300
    step_size = .1

    # plotting grid
    res = 200
    Xs_q = np.linspace(-20, 20, res)
    Ys_q = np.linspace(-10, 10, res)
    Xs_p = np.linspace(-1, 1, res)
    Ys_p = np.linspace(-1, 1, res)

    # evaluate density and estimate
    G = evaluate_density_grid(Xs_q, Ys_q, logq)
    G_est = evaluate_density_grid(Xs_q, Ys_q, logq_est)

    # evaluate momentum, which is the same for both
    M = evaluate_density_grid(Xs_p, Ys_p, logp)

    # simulate true and approximate Hamiltonian
    Qs, Ps = leapfrog(q0, dlogq, p0, dlogp, step_size, num_steps)
    Qs_est, Ps_est = leapfrog(q0, dlogq_est, p0, dlogp, step_size, num_steps)
    Hs = compute_hamiltonian(Qs, Ps, logq, logp)
    Hs_est = compute_hamiltonian(Qs_est, Ps_est, logq, logp)
    
    # normalise Hamiltonians
    Hs -= Hs.mean()
    Hs_est -= Hs_est.mean()
    
    plt.figure()
    plot_array(Xs_q, Ys_q, np.exp(G))
    plot_2d_trajectory(Qs)
    plt.title("HMC")
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.savefig(fname_base + "_hmc.eps", axis_inches="tight")
    
    plt.figure()
    plot_array(Xs_q, Ys_q, np.exp(G_est))
    plt.plot(Z[:, 0], Z[:, 1], 'bx')
    plot_2d_trajectory(Qs_est)
    plt.title("KMC")
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.savefig(fname_base + "_kmc.eps", axis_inches="tight")
    
    plt.figure()
    plt.title("Momentum")
    plot_array(Xs_p, Ys_p, np.exp(M))
    plot_2d_trajectory(Ps)
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.savefig(fname_base + "_momentum_hmc.eps", axis_inches="tight")
    
    plt.figure()
    plt.title("Momentum")
    plot_array(Xs_p, Ys_p, np.exp(M))
    plot_2d_trajectory(Ps_est)
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.savefig(fname_base + "_momentum_kmc.eps", axis_inches="tight")
    
    ylim = [np.min([Hs.min(), Hs_est.min()]),
            np.max([Hs.max(), Hs_est.max()])]
    
    
    plt.figure()
    plt.title("Hamiltonian")
    plt.plot(Hs)
    plt.ylim(ylim)
    plt.gca().xaxis.set_visible(False)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel("Leap-frog step")
    plt.ylabel(r"$H(p,q)$")
    plt.savefig(fname_base + "_hamiltonian_hmc.eps", axis_inches="tight")
    
    plt.figure()
    plt.title("Hamiltonian")
    plt.plot(Hs_est)
    plt.ylim(ylim)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().xaxis.set_visible(False)
    plt.xlabel("Leap-frog step")
    plt.ylabel(r"$H(p,q)$")
    plt.savefig(fname_base + "_hamiltonian_kmc.eps", axis_inches="tight")
    
    plt.show()
