from abc import abstractmethod
import os

from kmc.hamiltonian.hamiltonian import compute_log_accept_pr
from kmc.hamiltonian.leapfrog import leapfrog, leapfrog_no_storing
from kmc.score_matching.random_feats.estimator import RandomFeatsEstimator
from kmc.score_matching.random_feats.gaussian_rkhs import sample_basis, \
    score_matching_sym
from kmc.score_matching.random_feats.gaussian_rkhs_xvalidation import select_sigma_lambda_cma
from kmc.tools.Log import logger
import numpy as np
from scripts.experiments.mcmc.independent_job_classes.HMCJob import HMCJob,\
    HMCJobResultAggregator
from scripts.tools.plotting import evaluate_gradient_grid, plot_array,\
    evaluate_density_grid


splitted = __file__.split(os.sep)
idx = splitted.index('kamiltonian')
project_path = os.sep.join(splitted[:(idx + 1)])

temp = 0

class KMCRandomFeatsJob(HMCJob):
    def __init__(self, Z, m, sigma, lmbda,
                 target, momentum,
                 num_iterations, start,
                 num_steps_min=10, num_steps_max=100, step_size_min=0.05,
                 step_size_max=0.3, momentum_seed=0,
                 learn_parameters=False, force_relearn_parameters=False,
                 statistics={}, num_warmup=500, thin_step=1):
        
        HMCJob.__init__(self, target, momentum,
                        num_iterations, start,
                        num_steps_min, num_steps_max, step_size_min,
                        step_size_max, momentum_seed, statistics, num_warmup, thin_step)
        
        self.aggregator = KMCJobResultAggregator()
        
        self.Z = Z
        self.m = m
        self.sigma = sigma
        self.lmbda = lmbda
        self.learn_parameters = learn_parameters
        self.force_relearn_parameters = force_relearn_parameters
        
        self.upper_bound_N = 2000
        self.plot = False

    @abstractmethod
    def set_up(self):
        if self.learn_parameters or self.force_relearn_parameters:
            self.sigma, self.lmbda = self.determine_sigma_lmbda()
        
        logger.info("Using sigma=%.2f, lmbda=%.6f" % (self.sigma, self.lmbda))
        
        
        gamma = 0.5 * (self.sigma ** 2)
        logger.info("Sampling random basis")
        omega, u = sample_basis(self.D, self.m, gamma)
        
        logger.info("Estimating density in RKHS, N=%d, m=%d, D=%d" % (len(self.Z), self.m, self.D))
        theta = score_matching_sym(self.Z, self.lmbda, omega, u)
        
        # replace target by kernel estimator to simulate trajectories on
        # but keep original target for computing acceptance probability
        self.orig_target = self.target
        self.target = RandomFeatsEstimator(theta, omega, u)
        
        HMCJob.set_up(self)
        
        # plot density estimate
        if self.plot:
            import matplotlib.pyplot as plt
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
    def accept_prob_log_pdf(self, current, q, p0_log_pdf, p_log_pdf, current_log_pdf=None, samples=None):
        # potentially re-use log_pdf of last accepted state
        if current_log_pdf is None:
            current_log_pdf = -np.inf
        
        # same as super-class, but with original target
        kernel_target = self.target
        self.target = self.orig_target
        
        acc_prob, log_pdf_q = HMCJob.accept_prob_log_pdf(self, current, q, p0_log_pdf, p_log_pdf, current_log_pdf)
        
        # restore target
        self.target = kernel_target
        
        return acc_prob, log_pdf_q
    
    @abstractmethod
    def get_parameter_fname_suffix(self):
        return ("KMC_N=%d_" % len(self.Z)) + HMCJob.get_parameter_fname_suffix(self)[4:] 
    
    def determine_sigma_lmbda(self):
        parameter_dir = project_path + os.sep + "xvalidation_parameters"
        fname = parameter_dir + os.sep + self.get_parameter_fname_suffix() + ".npy"
        
        # upper bound for doing x-validation
        if len(self.Z) > self.upper_bound_N:
            fname.replace("N=%d" % self.D, "N=%d" % self.upper_bound_N)
        
        if not os.path.exists(fname) or self.force_relearn_parameters:
            logger.info("Learning sigma and lmbda")
            cma_opts = {'tolfun':0.3, 'maxiter':10, 'verb_disp':1}
            num_threads = 1 if self.force_relearn_parameters else 6
            sigma, lmbda = select_sigma_lambda_cma(self.Z, len(self.Z),
                                                   sigma0=self.sigma, lmbda0=self.lmbda,
                                                   cma_opts=cma_opts, num_threads=num_threads)
            
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
    
    @staticmethod
    def result_dict_from_file(fname):
        """
        Assumes a file with lots of lines as the one created by
        store_fire_and_forget_result and produces a dictionary with (N,D) as key
        arrays with experimental results for each of the R repetitions. This contains
        a few standard values as acceptance probability and the used posterior statistics
        """
        results = np.loadtxt(fname)
        
        result_dict = {}
        for i in range(len(results)):
            N = np.int(results[i, 0])
            D = np.int(results[i, 1])
            
            result_dict[(N, D)] = []
    
        for i in range(len(results)):
            N = np.int(results[i, 0])
            D = np.int(results[i, 1])
            time_taken_set_up = np.int(results[i, 2])
            time_taken_sampling = np.int(results[i, 3])
            accepted = np.float(results[i, 4])
            avg_quantile_error = results[i, 5]
            avg_ESS = results[i, 6]
            norm_of_mean = results[i, 7]
            
            to_add = np.zeros(6)
            to_add[0] = time_taken_set_up
            to_add[1] = time_taken_sampling
            to_add[2] = accepted
            to_add[3] = avg_quantile_error
            to_add[4] = avg_ESS
            to_add[5] = norm_of_mean
            
            result_dict[(N, D)] += [to_add]
        
        for k,v in result_dict.items():
            result_dict[k] = np.array(v)
        
        return result_dict


class KMCJobResultAggregator(HMCJobResultAggregator):
    def __init__(self):
        HMCJobResultAggregator.__init__(self)
        
    @abstractmethod
    def fire_and_forget_result_strings(self):
        strings = HMCJobResultAggregator.fire_and_forget_result_strings(self)
        
        return [str(self.result.D)] + strings
