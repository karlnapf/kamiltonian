from os import makedirs
import os
from os.path import expanduser
import time
import uuid

from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.FileSystem import FileSystem

from kmc.densities.banana import Banana, sample_banana
import kmc.densities.banana
from kmc.densities.gaussian import IsotropicZeroMeanGaussian
from kmc.tools.Log import logger
from kmc.tools.convergence_stats import ess_coda
import numpy as np
from scripts.experiments.mcmc.independent_job_classes.HMCJob import HMCJob
from scripts.experiments.mcmc.independent_job_classes.KMCRandomFeatsJob import KMCRandomFeatsJob
from scripts.experiments.mcmc.independent_job_classes.debug import plot_diagnosis


modulename = __file__.split(os.sep)[-1].split('.')[-2]
start_base = [0, -3.]

def hmc_generator(D, target, num_warmup, thin_step):
    momentum = IsotropicZeroMeanGaussian(sigma=sigma_p, D=D)
    start = np.array(start_base + [0. ] * (D - 2))
    
    return HMCJob(target, momentum, num_iterations, start,
                         num_steps_min, num_steps_max, step_size_min, step_size_max,
                         momentum_seed, statistics={"emp_quantiles": kmc.densities.banana.emp_quantiles,
                                                    "ESS": ess_coda},
                         num_warmup=num_warmup, thin_step=thin_step)

def kmc_generator(N, D, target, num_warmup, thin_step):
    momentum = IsotropicZeroMeanGaussian(sigma=sigma_p, D=D)
    start = np.array(start_base + [0. ] * (D - 2))
    
    # estimator parameters
    sigma = 0.46
    lmbda = 0.000001

    learn_parameters = True if N < 500 else False
    force_relearn_parameters = True if N < 500 else False

    # oracle samples
    Z = sample_banana(N, D, bananicity, V)
    job = KMCRandomFeatsJob(Z, sigma, lmbda,
                            target, momentum, num_iterations,
                            start, num_steps_min, num_steps_max,
                            step_size_min, step_size_max, momentum_seed, learn_parameters=learn_parameters,
                            force_relearn_parameters=force_relearn_parameters,
                            statistics={"emp_quantiles": kmc.densities.banana.emp_quantiles,
                                        "ESS": ess_coda},
                            num_warmup=num_warmup, thin_step=thin_step)
    job.plot = False
    return job

if __name__ == "__main__":
    logger.setLevel(10)
    Ds = np.sort([2, 8])[::-1]
    Ns = np.sort([10, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])[::-1]
#     Ds = [8]
#     Ns = [1000]
    
    
    print(Ns)
    print(Ds)
    assert np.min(Ds) >= 2
    num_repetitions = 10
#     num_repetitions = 1
    
    
    
    # target
    bananicity = 0.03
    V = 100
    target = Banana(bananicity, V)
    
    # plain MCMC parameters
    num_warmup = 500
    thin_step = 1
    num_iterations = 2000 + num_warmup
    
    # hmc parameters
    num_steps_min = 10
    num_steps_max = 100
    step_size_min = 0.05
    step_size_max = 0.3
    sigma_p = 1.
    momentum_seed = np.random.randint(time.time())

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
    
    aggs = {}
    for N in Ns:
        for D in Ds:
            aggs[D] = []
            aggs[(N, D)] = []
            
            
    for i in range(num_repetitions):
        momentum_seed += i
        for D in Ds:
            job = hmc_generator(D, target, num_warmup, thin_step)
            logger.info("Repetition %d/%d, %s" % (i + 1, num_repetitions, job.get_parameter_fname_suffix()))
            aggs[D] += [engine.submit_job(job)]
            for N in Ns:
                job = kmc_generator(N, D, target, num_warmup, thin_step)
                logger.info("Repetition %d/%d, %s" % (i + 1, num_repetitions, job.get_parameter_fname_suffix()))
                aggs[(N, D)] += [engine.submit_job(job)]
    
    engine.wait_for_all()
    
    if isinstance(engine, SerialComputationEngine):
        directory = expanduser("~") + os.sep + modulename
        try:
            makedirs(directory)
        except OSError:
            pass
        for D in Ds:
            for agg in aggs[D]:
                job_name = unicode(uuid.uuid4())
                agg.store_fire_and_forget_result(directory, job_name)
                
            for N in Ns:
                for agg in aggs[(N, D)]:
                    job_name = unicode(uuid.uuid4())
                    agg.store_fire_and_forget_result(directory, job_name)

        # plot some diagnosis
        
        for D in Ds:
            for agg in aggs[D]:
                plot_diagnosis(agg, D)
                
            for N in Ns:
                for agg in aggs[(N, D)]:
                    plot_diagnosis(agg, D)
