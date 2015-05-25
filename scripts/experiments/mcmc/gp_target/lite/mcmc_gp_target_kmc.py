from collections import OrderedDict
import os
import time

from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.FileSystem import FileSystem

from kmc.densities.gaussian import IsotropicZeroMeanGaussian
from kmc.densities.gp_classification_posterior_ard import GlassPosterior
from kmc.score_matching.lite.gaussian_rkhs_xvalidation import select_sigma_grid
from kmc.tools.Log import logger
from kmc.tools.convergence_stats import avg_ess, min_ess
import numpy as np
from scripts.experiments.mcmc.independent_job_classes.KMCLiteJob import KMCLiteJob
from scripts.experiments.mcmc.independent_job_classes.MCMCJob import MCMCJobResultAggregatorStoreHome
from scripts.experiments.mcmc.independent_job_classes.debug import plot_mcmc_result


modulename = __file__.split(os.sep)[-1].split('.')[-2]
statistics = OrderedDict()
statistics['avg_ess'] = avg_ess
statistics['min_ess'] = min_ess

# vanishing adaptation schedule as a function of the MCMC iteration
def schedule(i):
    start = 1000
    warmup_end = 1000
    decay = 0.2
    
    if i <= start:
        # just run for a while
        return 0.
    elif i <= warmup_end:
        # dont start schedule yet
        return 1.
    else:
        effective_it = i - warmup_end
        return 1. / effective_it ** decay

def non_adapt_schedule(i):
    return 0.

def kmc_generator(num_warmup, thin_step):
    D = 9
    start = np.random.randn(D) * 0
    
    step_size_min = 0.01
    step_size_max = 0.1
    num_steps_min = 1
    num_steps_max = 10
    sigma_p = 1.
    
    momentum_seed = np.random.randint(time.time())
    
    momentum = IsotropicZeroMeanGaussian(sigma=sigma_p, D=D)
    
    target = GlassPosterior()
    Z = np.load("../ground_truth/benchmark_samples.arr")[:1000]
    learn_parameters = False
    force_relearn_parameters = False
    
    lmbda = 1.
    if False:
        sigma = select_sigma_grid(Z, lmbda=lmbda, plot_surface=True)

    sigma = 2 ** 4
    
    job = KMCLiteJob(Z, sigma, lmbda, target,
                            momentum, num_iterations,
                            start,
                            num_steps_min, num_steps_max, step_size_min, step_size_max,
                            momentum_seed, learn_parameters, force_relearn_parameters,
                            statistics, num_warmup, thin_step)
    
    job.walltime = 60 * 60
    
    # store results in home dir straight away
    d = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1]) + os.sep
    job.aggregator = MCMCJobResultAggregatorStoreHome(d)
    
    return job


if __name__ == "__main__":
    logger.setLevel(10)
    num_repetitions = 10
    
    # plain MCMC parameters, plan is to use every 200th sample
    thin_step = 1
    num_iterations = 5200
    num_warmup = 200
    
    compute_local = False
    
    if not FileSystem.cmd_exists("sbatch") or compute_local:
        engine = SerialComputationEngine()
        
    else:
        johns_slurm_hack = "#SBATCH --partition=intel-ivy,wrkstn,compute"
        folder = os.sep + os.sep.join(["nfs", "data3", "ucabhst", modulename])
        batch_parameters = BatchClusterParameters(foldername=folder,
                                                  resubmit_on_timeout=False,
                                                  parameter_prefix=johns_slurm_hack)
        engine = SlurmComputationEngine(batch_parameters, check_interval=1,
                                        do_clean_up=True)
        engine.max_jobs_in_queue = 1000
        engine.store_fire_and_forget = True
    
    aggs = []
    
    for i in range(num_repetitions):
        job = kmc_generator(num_warmup, thin_step)
        logger.info("Repetition %d/%d, %s" % (i + 1, num_repetitions, job.get_parameter_fname_suffix()))
        aggs += [engine.submit_job(job)]
        
    engine.wait_for_all()
    
    for i, agg in enumerate(aggs):
        agg.finalize()
        result = agg.get_final_result()
        agg.clean_up()

        if isinstance(engine, SerialComputationEngine):
            plot_mcmc_result(result, D1=1, D2=6)
            agg.store_fire_and_forget_result(folder="", job_name="")
            
        
        # print some summary stats
        accepted = result.accepted
        avg_ess = result.posterior_statistics["avg_ess"]
        
        time = result.time_taken_sampling
        time_set_up = result.time_taken_set_up
    
        logger.info("Repetition %d" % i)
        logger.info("Average acceptance probability: %.2f" % np.mean(accepted))
        logger.info("Average ESS: %.2f" % avg_ess)
        logger.info("Total time: %.2f" % (time + time_set_up))
