from abc import abstractmethod
import os

from independent_jobs.jobs.IndependentJob import IndependentJob

from kmc.densities.gaussian import sample_gaussian, log_gaussian_pdf
from kmc.hamiltonian.hamiltonian import compute_log_accept_pr, \
    compute_log_det_trajectory
from kmc.hamiltonian.leapfrog import leapfrog
from kmc.score_matching.random_feats.estimator import log_pdf_estimate_grad
from kmc.score_matching.random_feats.gaussian_rkhs import score_matching_sym, \
    sample_basis, feature_map_grad_single
from kmc.score_matching.random_feats.gaussian_rkhs_xvalidation import select_sigma_lambda_cma
from kmc.tools.Log import logger
import numpy as np
from scripts.experiments.trajectories.independent_jobs_classes.TrajectoryJobResult import TrajectoryJobResult
from scripts.experiments.trajectories.independent_jobs_classes.TrajectoryJobResultAggregator import TrajectoryJobResultAggregator


splitted = __file__.split(os.sep)
idx = splitted.index('kamiltonian')
project_path = os.sep.join(splitted[:(idx+1)])

class TrajectoryJob(IndependentJob):
    def __init__(self,
                 N, D, m, sigma_p, num_steps,
                 step_size, max_steps=None,
                 sigma0=0.5, lmbda0=0.0001):
        IndependentJob.__init__(self, TrajectoryJobResultAggregator())
        
        # job ressources
        if N <= 2000 or D < 10:
            self.walltime = 1 * 60 * 60
            self.memory = 2
        elif N <= 5000 or D < 50:
            self.walltime = 24 * 60 * 60
            self.memory = 3
        else:
            self.walltime = 72 * 60 * 60
            self.memory = 8
        
        self.N = N
        self.D = D
        self.m = m
        self.sigma0 = sigma0
        self.lmbda0 = lmbda0
        self.sigma_p = sigma_p
        self.num_steps = num_steps
        self.step_size = step_size
        self.max_steps = max_steps
    
    @abstractmethod
    def set_up(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_parameter_fname_suffix(self):
        # for N>2000, use m=2000
        N = m = np.min([self.N, 2000])
        return "N=%d_m=%d_D=%d" % (N, m, self.D)
    
    def determine_sigma_lmbda(self):
        return 0.5, 0.00008
    
    def _determine_sigma_lmbda(self):
        parameter_dir = project_path + os.sep + "xvalidation_parameters"
        fname = parameter_dir + os.sep + self.get_parameter_fname_suffix() + ".npy"
        if not os.path.exists(fname):
            logger.info("Learning sigma and lmbda")
            cma_opts = {'tolfun':0.5, 'maxiter':10, 'verb_disp':1}
            sigma, lmbda = select_sigma_lambda_cma(self.Z, self.m,
                                                   sigma0=self.sigma0, lmbda0=self.lmbda0,
                                                   cma_opts=cma_opts)
            
            if not os.path.exists(parameter_dir):
                os.makedirs(parameter_dir)
            
            with open(fname, 'w+') as f:
                np.savez_compressed(f, sigma=sigma, lmbda=lmbda)
        else:
            logger.info("Loading sigma and lmbda from %s" % fname)
            with open(fname, 'r') as f:
                pars = np.load(f)
                sigma = pars['sigma']
                lmbda = pars['lmbda']
                
        return sigma, lmbda
            
    def compute_trajectory(self, random_start_state=None):
        logger.debug("Entering")
        
        if random_start_state is not None:
            np.random.set_state(random_start_state)
        else:
            random_start_state = np.random.get_state()
        
        # momentum
        L_p = np.linalg.cholesky(np.eye(self.D) * self.sigma_p)
        self.logp = lambda x: log_gaussian_pdf(x, Sigma=L_p, compute_grad=False, is_cholesky=True)
        self.dlogp = lambda x: log_gaussian_pdf(x, Sigma=L_p, compute_grad=True, is_cholesky=True)
        self.p_sample = lambda: sample_gaussian(N=1, mu=np.zeros(self.D), Sigma=L_p, is_cholesky=True)[0]
        self.p_sample = lambda: sample_gaussian(N=1, mu=np.zeros(self.D), Sigma=L_p, is_cholesky=True)[0]
        
        # set up target and momentum densities and gradients
        self.set_up()
        
        # load or learn parameters
        sigma, lmbda = self.determine_sigma_lmbda()
        
        logger.info("Using sigma: %.2f, lmbda=%.6f" % (sigma, lmbda))
        
        D = self.Z.shape[1]
        gamma = 0.5 * (sigma ** 2)
        omega, u = sample_basis(D, self.m, gamma)
        
        logger.info("Estimate density in RKHS, N=%d, m=%d" % (self.N, self.m))
        theta = score_matching_sym(self.Z, lmbda, omega, u)
        
#         logger.info("Computing objective function")
#         J = _objective_sym(Z, sigma, lmbda, a, K, b, C)
#         J_xval = np.mean(xvalidate(Z, 5, sigma, self.lmbda, K))
#         logger.info("N=%d, sigma: %.2f, lambda: %.2f, J(a)=%.2f, XJ(a)=%.2f" % \
#                 (self.N, sigma, self.lmbda, J, J_xval))
        
        dlogq_est = lambda x: log_pdf_estimate_grad(feature_map_grad_single(x, omega, u),
                                                    theta)
        
        # random number of steps?
        if self.max_steps is not None:
            steps = np.random.randint(self.num_steps, self.max_steps + 1)
        else:
            steps = self.num_steps
        
        logger.info("Simulating trajectory for at least L=%d steps of size %.2f" % \
                     (self.num_steps, self.step_size))
        # starting state
        p0 = self.p_sample()
        q0 = self.q_sample()
        
        Qs, Ps = leapfrog(q0, self.dlogq, p0, self.dlogp, self.step_size, steps)
        
        # run second integrator for same amount of steps
        steps_taken = len(Qs)
        Qs_est, Ps_est = leapfrog(q0, dlogq_est, p0, self.dlogp, self.step_size, steps_taken)
        logger.info("%d steps taken" % steps_taken)
        
        logger.info("Computing average acceptance probabilities")
        log_acc = compute_log_accept_pr(q0, p0, Qs, Ps, self.logq, self.logp)
        log_acc_est = compute_log_accept_pr(q0, p0, Qs_est, Ps_est, self.logq, self.logp)
        acc_mean = np.mean(np.exp(log_acc))
        acc_est_mean = np.mean(np.exp(log_acc_est))
        idx09 = int(len(log_acc) * 0.9)
        acc_mean10 = np.mean(np.exp(log_acc[idx09:]))
        acc_est_mean10 = np.mean(np.exp(log_acc_est[idx09:]))
        
        logger.info("Computing average volumes")
        log_det = compute_log_det_trajectory(Qs, Ps)
        log_det_est = compute_log_det_trajectory(Qs_est, Ps_est)
        
        logger.info("Average acceptance prob: %.2f, %.2f" % (acc_mean, acc_est_mean))
        logger.info("Average acceptance prob (last 10 percent): %.2f, %.2f" % (acc_mean10, acc_est_mean10))
        logger.info("Log-determinant: %.2f, %.2f" % (log_det, log_det_est))
        
        logger.debug("Leaving")
        return acc_mean, acc_est_mean, log_det, log_det_est, steps_taken, random_start_state
    
    def compute(self):
        logger.debug("Entering")
        random_start_state = np.random.get_state()
        
        acc_mean, acc_est_mean, log_det, log_det_est, steps_taken, random_start_state = \
            self.compute_trajectory(random_start_state)
        
        logger.info("Submitting results to aggregator")
        result = TrajectoryJobResult(self.D, self.N, acc_mean, acc_est_mean, log_det,
                                     log_det_est, steps_taken, random_start_state)
        self.aggregator.submit_result(result)
        
        logger.debug("Leaving")
