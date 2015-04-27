from abc import abstractmethod
import os

from kmc.score_matching.random_feats.estimator import RandomFeatsEstimator
from kmc.score_matching.random_feats.gaussian_rkhs import sample_basis, \
    score_matching_sym
from kmc.score_matching.random_feats.gaussian_rkhs_xvalidation import select_sigma_lambda_cma
from kmc.tools.Log import logger
from scripts.experiments.mcmc.independent_job_classes.HMCJob import HMCJob,\
    HMCJobResultAggregator
    
import numpy as np


splitted = __file__.split(os.sep)
idx = splitted.index('kamiltonian')
project_path = os.sep.join(splitted[:(idx+1)])

class KMCRandomFeatsJob(HMCJob):
    def __init__(self, Z, sigma, lmbda,
                 target, momentum,
                 num_iterations, start,
                 num_steps_min=10, num_steps_max=100, step_size_min=0.05,
                 step_size_max=0.3, momentum_seed=0,
                 learn_parameters=False,
                 statistics = {}, num_warmup=500, thin_step=1):
        
        HMCJob.__init__(self, target, momentum,
                        num_iterations, start,
                        num_steps_min, num_steps_max, step_size_min,
                        step_size_max, momentum_seed, statistics, num_warmup, thin_step)
        
        self.aggregator = KMCJobResultAggregator()
        
        self.Z = Z
        self.sigma = sigma
        self.lmbda = lmbda
        self.learn_parameters = learn_parameters

    @abstractmethod
    def set_up(self):
        # match number of basis functions and data
        m = len(self.Z)

        if self.learn_parameters:
            logger.info("Learning parameters")
            self.sigma, self.lmbda = self.determine_sigma_lmbda()
        
        logger.info("Using sigma=%.2f, lmbda=%.6f" % (self.sigma, self.lmbda))
        
        
        gamma = 0.5 * (self.sigma ** 2)
        logger.info("Sampling random basis")
        omega, u = sample_basis(self.D, m, gamma)
        
        logger.info("Estimating density in RKHS")
        theta = score_matching_sym(self.Z, self.lmbda, omega, u)
        
        # replace target by kernel estimator to simulate trajectories on
        # but keep original target for computing acceptance probability
        self.orig_target = self.target
        self.target = RandomFeatsEstimator(theta, omega, u)
        
        HMCJob.set_up(self)
        
        # plot density estimate
        if self.plot:
            import matplotlib.pyplot as plt
            import numpy as np
            from scripts.tools.plotting import evaluate_density_grid, evaluate_gradient_grid, plot_array
            
            Xs = np.linspace(-15, 15)
            Ys = np.linspace(-7, 3)
            Xs_grad = np.linspace(-40, 40, 40)
            Ys_grad = np.linspace(-15, 25, 40)
            G = evaluate_density_grid(Xs, Ys, self.target.log_pdf)
            G_norm, quiver_U, quiver_V, _, _ = evaluate_gradient_grid(Xs_grad, Ys_grad, self.target.grad)
            plt.subplot(211)
            plt.plot(self.Z[:, 0], self.Z[:, 1], 'bx')
            plot_array(Xs, Ys, np.exp(G), plot_contour=True)
            plt.subplot(212)
            plot_array(Xs_grad, Ys_grad, G_norm, plot_contour=True)
            plt.quiver(Xs_grad, Ys_grad, quiver_U, quiver_V, color='m')
            plt.ioff()
            plt.show()
        
    
    @abstractmethod
    def accept_prob_log_pdf(self, current, q, p0_log_pdf, p_log_pdf, current_log_pdf=None):
        # same as super-class, but with original target
        kernel_target = self.target
        self.target = self.orig_target
        
        acc_prob, log_pdf_q = HMCJob.accept_prob_log_pdf(self, current, q, p0_log_pdf, p_log_pdf, current_log_pdf)
        
        # restore target
        self.target = kernel_target
        
        return acc_prob, log_pdf_q
    
    @abstractmethod
    def get_parameter_fname_suffix(self):
        return ("KMC_N=%d_" % len(self.Z)) + HMCJob.get_parameter_fname_suffix(self) 
    
    def determine_sigma_lmbda(self):
        parameter_dir = project_path + os.sep + "xvalidation_parameters"
        fname = parameter_dir + os.sep + self.get_parameter_fname_suffix() + ".npy"
        if not os.path.exists(fname) or True:
            logger.info("Learning sigma and lmbda")
            cma_opts = {'tolfun':0.3, 'maxiter':10, 'verb_disp':1}
            sigma, lmbda = select_sigma_lambda_cma(self.Z, len(self.Z),
                                                   sigma0=self.sigma, lmbda0=self.lmbda,
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


class KMCJobResultAggregator(HMCJobResultAggregator):
    def __init__(self):
        HMCJobResultAggregator.__init__(self)
        
    @abstractmethod
    def fire_and_forget_result_strings(self):
        strings = HMCJobResultAggregator.fire_and_forget_result_strings(self)
        
        for i in range(len(strings)):
            strings[i] = ("%d " % len(self.result.mcmc_job.Z)) + strings[i]
        
        return strings