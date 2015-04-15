from kmc.densities.gaussian import log_gaussian_pdf, sample_gaussian
from kmc.densities.student import log_student_pdf
from kmc.score_matching.random_feats.estimator import log_pdf_estimate, \
    log_pdf_estimate_grad
from kmc.score_matching.random_feats.gaussian_rkhs import feature_map, \
    score_matching_sym, feature_map_grad_single
from kmc.tools.Log import logger
import matplotlib.pyplot as plt
import numpy as np
from scripts.tools.plotting import plot_kamiltonian_dnyamics


logger.setLevel(10)

# if __name__ == "__main__":
while True:
    D = 2
    
    # true target log density
    nu = 1.  # degrees of freedom
    dlogq = lambda x: log_student_pdf(x, nu, True)
    logq = lambda x: log_student_pdf(x, nu, False)

    # estimate density in rkhs
    N = 800
    mu = np.zeros(D)
    Z = np.random.standard_t(df=nu, size=(N, D))
    lmbda = 0.0001
    sigma = 0.5
    gamma = 0.5 * (sigma ** 2)
    m = N
    
    omega = gamma * np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    logger.info("Estimating density")
    theta = score_matching_sym(Z, lmbda, omega, u)
    
    logq_est = lambda x: log_pdf_estimate(feature_map(x, omega, u), theta)
    dlogq_est = lambda x: log_pdf_estimate_grad(feature_map_grad_single(x, omega, u), theta)
    
    # momentum
    Sigma_p = np.eye(D)
    L_p = np.linalg.cholesky(Sigma_p)
    logp = lambda x: log_gaussian_pdf(x, Sigma=L_p, compute_grad=False, is_cholesky=True)
    dlogp = lambda x: log_gaussian_pdf(x, Sigma=L_p, compute_grad=True, is_cholesky=True)
    p_sample = lambda: sample_gaussian(N=1, mu=np.zeros(D), Sigma=L_p, is_cholesky=True)[0]

    # starting state
    p0 = p_sample()
    q0 = np.zeros(D)
    q0[:2] = np.array([-1, -1])
    
    # parameters
    num_steps = 500
    step_size = .1
    
    Xs_q = np.linspace(-3, 3)
    Ys_q = np.linspace(-3, 3)
    Xs_p = np.linspace(-3, 3)
    Ys_p = np.linspace(-3, 3)

    logger.info("Evaluating dynamics")
    plot_grad_target = False
    plot_kamiltonian_dnyamics(q0, p0,
                              logq, dlogq, logq_est, dlogq_est, logp, dlogp, Z,
                              num_steps, step_size,
                              Xs_q, Ys_q, Xs_p, Ys_p, plot_dlogq=plot_grad_target,
                              plot_H_or_acc=False)
#     plt.suptitle(r"Score match, $J(\alpha)=%.2f$, $\lambda=%.2f$, $\sigma=%.2f$" % (J, lmbda, sigma))
    plt.show()
