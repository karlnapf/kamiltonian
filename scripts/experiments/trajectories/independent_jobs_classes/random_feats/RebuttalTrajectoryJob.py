from abc import abstractmethod

from kmc.densities.gaussian import log_gaussian_pdf, sample_gaussian
from kmc.score_matching.random_feats.estimator import log_pdf_estimate_grad
from kmc.score_matching.random_feats.gaussian_rkhs import sample_basis_rational_quadratic,\
    score_matching_sym, feature_map_grad_single
from kmc.tools.Log import logger
from kmc.tools.numerics import qmult
import numpy as np
from scripts.experiments.trajectories.independent_jobs_classes.random_feats.TrajectoryJob import TrajectoryJob


class RebuttalTrajectoryJob(TrajectoryJob):
    """
    Note that sigma0 will be used as alpha for the RQ kernel, with beta=1/ell^2 = 1
    """
    def __init__(self,
                 N, D, m,
                 sigma_p,
                 num_steps, step_size, sigma0=0.5, lmbda0=0.0001, max_steps=None,
                 learn_parameters=False):
        TrajectoryJob.__init__(self, N, D, m, sigma_p,
                               num_steps, step_size, max_steps, sigma0, lmbda0,
                               learn_parameters=learn_parameters)
        
    @abstractmethod
    def set_up(self):
        # place a gamma on the Eigenvalues of a Gaussian covariance
        EVs = np.random.gamma(shape=1, size=self.D)
        
        # random orthogonal matrix to rotate
        Q = qmult(np.eye(self.D))
        Sigma = Q.T.dot(np.diag(EVs)).dot(Q)
        
        # Cholesky of random covariance
        L = np.linalg.cholesky(Sigma)
        
        # target density
        self.dlogq = lambda x: log_gaussian_pdf(x, Sigma=L, is_cholesky=True, compute_grad=True)
        self.logq = lambda x: log_gaussian_pdf(x, Sigma=L, is_cholesky=True, compute_grad=False)
    
        # starting state
        self.q_sample = lambda: sample_gaussian(N=1, mu=np.zeros(self.D), Sigma=L, is_cholesky=True)[0]
        
        logger.info("N=%d, D=%d" % (self.N, self.D))
        self.Z = sample_gaussian(self.N, mu=np.zeros(self.D), Sigma=L, is_cholesky=True)

    @abstractmethod
    def get_parameter_fname_suffix(self):
        suffix = self.__class__.__name__
        
        return suffix + "_" + TrajectoryJob.get_parameter_fname_suffix(self) 

    @abstractmethod
    def update_density_estimate(self):
        # load or learn parameters
        if self.learn_parameters:
            sigma, lmbda = self.determine_sigma_lmbda()
        else:
            sigma = self.sigma0
            lmbda = self.lmbda0
        
        logger.info("Using alpha: %.2f, lmbda=%.6f" % (sigma, lmbda))
        
        D = self.Z.shape[1]
        omega, u = sample_basis_rational_quadratic(D, self.m, alpha=sigma, beta=1.)
        
        logger.info("Estimate density in RKHS, N=%d, m=%d" % (self.N, self.m))
        theta = score_matching_sym(self.Z, lmbda, omega, u)
        
#         logger.info("Computing objective function")
#         J = _objective_sym(Z, sigma, lmbda, a, K, b, C)
#         J_xval = np.mean(xvalidate(Z, 5, sigma, self.lmbda, K))
#         logger.info("N=%d, sigma: %.2f, lambda: %.2f, J(a)=%.2f, XJ(a)=%.2f" % \
#                 (self.N, sigma, self.lmbda, J, J_xval))
        
        dlogq_est = lambda x: log_pdf_estimate_grad(feature_map_grad_single(x, omega, u),
                                                    theta)
        
        return dlogq_est
