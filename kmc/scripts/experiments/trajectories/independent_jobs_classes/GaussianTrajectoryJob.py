from independent_jobs.jobs.IndependentJob import IndependentJob

from kmc.densities.gaussian import sample_gaussian, log_gaussian_pdf
from kmc.hamiltonian.hamiltonian import compute_log_accept_pr,\
    compute_log_det_trajectory
from kmc.hamiltonian.leapfrog import leapfrog
from kmc.score_matching.estimator import log_pdf_estimate_grad
from kmc.score_matching.gaussian_rkhs import _compute_b_sym, _compute_C_sym,\
    score_matching_sym, _objective_sym, xvalidate
from kmc.score_matching.gaussian_rkhs_xvalidation import select_sigma_grid
from kmc.score_matching.kernel.kernels import gaussian_kernel,\
    gaussian_kernel_grad
from kmc.scripts.experiments.trajectories.independent_jobs_classes.TrajectoryJobResult import TrajectoryJobResult
from kmc.scripts.experiments.trajectories.independent_jobs_classes.TrajectoryJobResultAggregator import TrajectoryJobResultAggregator
from kmc.tools.Log import logger
from kmc.tools.numerics import log_mean_exp
import numpy as np


class GaussianTrajectoryJob(IndependentJob):
    def __init__(self,
                 N, mu, L, lmbda,
                 L_p,
                 num_steps, step_size):
        IndependentJob.__init__(self, TrajectoryJobResultAggregator())
        
        self.N = N
        self.mu = mu
        self.L = L
        self.lmbda = lmbda
        self.L_p = L_p
        self.num_steps = num_steps
        self.step_size = step_size
        
        # momentum
        D = L.shape[0]
        self.logp = lambda x: log_gaussian_pdf(x, Sigma=L_p, compute_grad=False, is_cholesky=True)
        self.dlogp = lambda x: log_gaussian_pdf(x, Sigma=L_p, compute_grad=True, is_cholesky=True)
        
        # target density
        self.dlogq = lambda x: log_gaussian_pdf(x, Sigma=L, is_cholesky=True, compute_grad=True)
        self.logq = lambda x: log_gaussian_pdf(x, Sigma=L, is_cholesky=True, compute_grad=False)
    
        # starting state
        self.q_sample = lambda: sample_gaussian(N=1, mu=np.zeros(D), Sigma=L, is_cholesky=True)[0]
        self.p_sample = lambda: sample_gaussian(N=1, mu=np.zeros(D), Sigma=L_p, is_cholesky=True)[0]
    
    def compute(self):
        logger.debug("Entering")
        
        logger.debug("Estimate density in RKHS")
        Z = sample_gaussian(self.N, self.mu, Sigma=self.L, is_cholesky=True)
        sigma = select_sigma_grid(Z, lmbda=self.lmbda, log2_sigma_max=15)
        
        K = gaussian_kernel(Z, sigma=sigma)
        b = _compute_b_sym(Z, K, sigma)
        C = _compute_C_sym(Z, K, sigma)
        a = score_matching_sym(Z, sigma, self.lmbda, K, b, C)
        J = _objective_sym(Z, sigma, self.lmbda, a, K, b, C)
        J_xval = np.mean(xvalidate(Z, 5, sigma, self.lmbda, K))
        print("N=%d, sigma: %.2f, lambda: %.2f, J(a)=%.2f, XJ(a)=%.2f" % \
                (self.N, sigma, self.lmbda, J, J_xval))
        
        kernel_grad = lambda x, X = None: gaussian_kernel_grad(x, X, sigma)
        dlogq_est = lambda x: log_pdf_estimate_grad(x, a, Z, kernel_grad)
        
        
        logger.debug("Simulating trajectory for L=%d steps of size %.2f" % \
                     (self.num_steps, self.step_size))
        # starting state
        p0 = self.p_sample()
        q0 = self.q_sample()
        
        Qs, Ps = leapfrog(q0, self.dlogq, p0, self.dlogp, self.step_size, self.num_steps)
        Qs_est, Ps_est = leapfrog(q0, dlogq_est, p0, self.dlogp, self.step_size, self.num_steps)
        
        logger.debug("Computing average acceptance probabilities")
        log_acc = compute_log_accept_pr(q0, p0, Qs, Ps, self.logq, self.logp)
        log_acc_est = compute_log_accept_pr(q0, p0, Qs_est, Ps_est, self.logq, self.logp)
        acc_mean = np.exp(log_mean_exp(log_acc))
        acc_est_mean = np.exp(log_mean_exp(log_acc_est))
        
        logger.debug("Computing average volumes")
        log_det = compute_log_det_trajectory(Qs, Ps)
        log_det_est = compute_log_det_trajectory(Qs_est, Ps_est)
        
        logger.debug("Submitting results to aggregator")
        result = TrajectoryJobResult(acc_mean, acc_est_mean, log_det, log_det_est)
        self.aggregator.submit_result(result)
        
        logger.debug("Leaving")
