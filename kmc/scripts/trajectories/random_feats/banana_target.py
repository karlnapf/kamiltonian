from kmc.densities.banana import log_banana_pdf, sample_banana
from kmc.densities.gaussian import log_gaussian_pdf, sample_gaussian
from kmc.hamiltonian.hamiltonian import compute_log_accept_pr
from kmc.score_matching.estimator import log_pdf_estimate, log_pdf_estimate_grad
from kmc.score_matching.gaussian_rkhs import _compute_b_sym, _compute_C_sym, \
    score_matching_sym, _objective_sym, xvalidate
from kmc.score_matching.gaussian_rkhs_xvalidation import select_sigma_grid
from kmc.score_matching.kernel.kernels import gaussian_kernel, \
    gaussian_kernel_grad
from kmc.scripts.tools.plotting import plot_kamiltonian_dnyamics
import matplotlib.pyplot as plt
import numpy as np


# if __name__ == "__main__":
while True:
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
    Sigma_p = np.eye(D)*.1
    L_p = np.linalg.cholesky(Sigma_p)
    logp = lambda x: log_gaussian_pdf(x, Sigma=L_p, compute_grad=False, is_cholesky=True)
    dlogp = lambda x: log_gaussian_pdf(x, Sigma=L_p, compute_grad=True, is_cholesky=True)
    p_sample = lambda: sample_gaussian(N=1, mu=np.zeros(D), Sigma=L_p, is_cholesky=True)[0]

    # starting state
    p0 = p_sample()
    q0 = np.zeros(D)
    q0[:2] = np.array([0, -3])
    
    # parameters
    num_steps = 1500
    step_size = .1

    Xs_q = np.linspace(-20, 20)
    Ys_q = np.linspace(-10, 10)
    Xs_p = np.linspace(-1, 1)
    Ys_p = np.linspace(-1, 1)

    plot_grad_target = False
    plot_kamiltonian_dnyamics(q0, p0,
                              logq, dlogq, logq_est, dlogq_est, logp, dlogp, Z,
                              num_steps, step_size,
                              Xs_q, Ys_q, Xs_p, Ys_p, plot_dlogq=plot_grad_target,
                              plot_H_or_acc=False)
    
    plt.suptitle(r"Score match, $J(\alpha)=%.2f$, $\lambda=%.2f$, $\sigma=%.2f$" % (J, lmbda, sigma))
    plt.show()
