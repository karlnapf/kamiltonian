from abc import abstractmethod
import os

from kmc.hamiltonian.hamiltonian import compute_log_accept_pr
from kmc.hamiltonian.leapfrog import leapfrog, leapfrog_no_storing
from kmc.score_matching.lite.estimator import LiteEstimatorGaussian
from kmc.score_matching.lite.gaussian_rkhs import score_matching_sym
from kmc.score_matching.random_feats.estimator import RandomFeatsEstimator
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

class KMCLiteJob(HMCJob):
    def __init__(self, Z, sigma, lmbda,
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
        self.sigma = sigma
        self.lmbda = lmbda
        self.learn_parameters = learn_parameters
        self.force_relearn_parameters = force_relearn_parameters
        
        self.upper_bound_N = 2000
        self.plot = False

    @abstractmethod
    def propose(self, current, current_log_pdf, samples, accepted):
        # random variables from a fixed random stream without modifying the current one
        rnd_state = np.random.get_state()
        np.random.set_state(self.hmc_rnd_state)
         
        if current_log_pdf is None:
            current_log_pdf = self.orig_target.log_pdf(current)
         
        # sample momentum and leapfrog parameters
        p0 = self.momentum.sample()
        num_steps = np.random.randint(self.num_steps_min, self.num_steps_max + 1)
        step_size = np.random.rand() * (self.step_size_max - self.step_size_min) + self.step_size_min
         
        # restore random state
        self.hmc_rnd_state = np.random.get_state()
        np.random.set_state(rnd_state)
         
        logger.debug("Simulating Hamiltonian flow")
        Qs, Ps = leapfrog(current, self.target.grad, p0, self.momentum.grad, step_size, num_steps)
         
        q=Qs[-1]
        p=Ps[-1]
         
        logger.debug("Momentum start: %s" % str(p0))
        logger.debug("Momentum end: %s" % str(p))
         
        # compute acceptance probability, extracting log_pdf of q
        p0_log_pdf = self.momentum.log_pdf(p0)
        p_log_pdf = self.momentum.log_pdf(p)
         
        # use a function call to be able to overload it for KMC
        acc_prob, log_pdf_q = self.accept_prob_log_pdf(current, q, p0_log_pdf, p_log_pdf, current_log_pdf, samples)
         
        if True and (len(samples) % 100) ==0:
            logger.debug("Plotting")
            import matplotlib.pyplot as plt
             
            res = 50
            Xs_q = np.linspace(-4,4, res)
            Ys_q = np.linspace(-4,4, res)
         
            # evaluate density and estimate
            D1=0
            D2=1
            def dummy_grad(X_2d):
                theta = current.copy()
#                 theta = np.mean(self.Z, 0)
                theta[D1]=X_2d[0]
                theta[D2]=X_2d[1]
                return self.target.grad(theta)
                 
            def dummy(X_2d):
                theta = current.copy()
#                 theta = np.mean(self.Z, 0)
                theta[D1]=X_2d[0]
                theta[D2]=X_2d[1]
                return self.target.log_pdf(theta)
             
#             plt.figure()
#             G = evaluate_density_grid(Xs_q, Ys_q, dummy)
#             plot_array(Xs_q, Ys_q, G)
#             plt.plot(self.Z[:,D1], self.Z[:,D2], '.')
#             plt.plot(Qs[:,D1], Qs[:,D2], 'r-')
#             plt.plot(samples[:,D1], samples[:,D2], 'm-')
#             plt.plot(current[D1], current[D2], 'b*', markersize=15)
#             plt.plot(Qs[-1,D1], Qs[-1,D2], 'r*', markersize=15)
             
            plt.figure()
            G_norm, U_q, V, X, Y = evaluate_gradient_grid(Xs_q, Ys_q, dummy_grad)
            plot_array(Xs_q, Ys_q, G_norm)
            plt.plot(self.Z[:,D1], self.Z[:,D2], '.')
            plt.plot(Qs[:,D1], Qs[:,D2], 'r-')
            plt.plot(samples[:,D1], samples[:,D2], 'm-')
            plt.plot(current[D1], current[D2], 'b*', markersize=15)
            plt.plot(Qs[-1,D1], Qs[-1,D2], 'r*', markersize=15)
            plt.quiver(X, Y, U_q, V, color='m')
             
#             plt.figure()
#             plt.plot(Ps[:,D1], Ps[:,D2], 'r-')
#             plt.plot(p0[D1], p0[D2], 'b*', markersize=15)
#             plt.plot(Ps[-1,D1], Ps[-1,D2], 'r*', markersize=15)
#             plt.title('momentum')
             
            acc_probs = np.exp(compute_log_accept_pr(current, p0, Qs, Ps, self.orig_target.log_pdf, self.momentum.log_pdf))
            H_ratios = np.exp(compute_log_accept_pr(current, p0, Qs, Ps, self.target.log_pdf, self.momentum.log_pdf))
            target_ratio = [np.min([1,np.exp(self.orig_target.log_pdf(x)-current_log_pdf)]) for x in Qs]
            momentum_ratio = [np.min([1,np.exp(self.momentum.log_pdf(x)-p0_log_pdf)]) for x in Ps]
            target_log_pdf = np.exp(np.array([self.orig_target.log_pdf(x) for x in Qs]))
# #              
#             plt.figure(figsize=(12,4))
#             plt.subplot(151)
#             plt.plot(acc_probs)
#             plt.plot([0, len(acc_probs)], [acc_probs.mean(), acc_probs.mean()])
#             plt.title("acc_probs")
#             plt.subplot(152)
#             plt.plot(target_ratio)
#             plt.title("target_ratio")
#             plt.subplot(153)
#             plt.plot(momentum_ratio)
#             plt.title("momentum_ratio")
#             plt.subplot(154)
#             plt.plot(H_ratios)
#             plt.title("H_ratios")
#             plt.subplot(155)
#             plt.plot(target_log_pdf)
#             plt.title("target_log_pdf")
             
             
             
            plt.show()
        
        return q, acc_prob, log_pdf_q

    @abstractmethod
    def set_up(self):
        if self.learn_parameters or self.force_relearn_parameters:
            self.sigma, self.lmbda = self.determine_sigma_lmbda()
        
        logger.info("Using sigma=%.2f, lmbda=%.6f" % (self.sigma, self.lmbda))
        
        
        logger.info("Estimating density in RKHS, N=%d,D=%d" % (len(self.Z), self.D))
        alpha = score_matching_sym(self.Z, self.sigma, self.lmbda)
        
        # replace target by kernel estimator to simulate trajectories on
        # but keep original target for computing acceptance probability
        self.orig_target = self.target
        self.target = LiteEstimatorGaussian(alpha, self.Z, self.sigma)
        
        HMCJob.set_up(self)
    
    @abstractmethod
    def accept_prob_log_pdf(self, current, q, p0_log_pdf, p_log_pdf, current_log_pdf, samples):
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
