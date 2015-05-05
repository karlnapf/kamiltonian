from os import makedirs
import os
from os.path import expanduser
import time
import uuid

from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.FileSystem import FileSystem

from kmc.densities.banana import Banana, sample_banana, norm_of_emp_mean
import kmc.densities.banana
from kmc.densities.gaussian import IsotropicZeroMeanGaussian
from kmc.tools.Log import logger
from kmc.tools.convergence_stats import avg_ess
import numpy as np
from scripts.experiments.mcmc.independent_job_classes.HMCJob import HMCJob
from scripts.experiments.mcmc.independent_job_classes.KMCRandomFeatsJob import KMCRandomFeatsJob
from scripts.experiments.mcmc.independent_job_classes.KameleonJob import KameleonJob
from scripts.experiments.mcmc.independent_job_classes.RWJob import RWJob
from scripts.experiments.mcmc.independent_job_classes.debug import plot_diagnosis


modulename = __file__.split(os.sep)[-1].split('.')[-2]
start_base = [0, -3.]

statistics = {"avg_quantile_error": kmc.densities.banana.avg_quantile_error,
                                    "avg_ess": avg_ess,
                                    "norm_of_mean": norm_of_emp_mean}

def hmc_generator(D, target, num_warmup, thin_step, momentum_seed):
    momentum = IsotropicZeroMeanGaussian(sigma=sigma_p, D=D)
    start = np.array(start_base + [0. ] * (D - 2))
    
    return HMCJob(target, momentum, num_iterations, start,
                  num_steps_min, num_steps_max, step_size_min, step_size_max, momentum_seed,
                  statistics=statistics, num_warmup=num_warmup, thin_step=thin_step)

def rw_generator_isotropic(D, target, num_warmup, thin_step):
    start = np.array(start_base + [0. ] * (D - 2))
    
    # tuned towards roughly 23% acceptance
    sigmas_proposal = {
              2: 4.4,
              8: 1.,
              16: 0.35,
              32: 0.05
              }
    
    return RWJob(target, num_iterations, start, sigmas_proposal[D],
                 statistics=statistics,
                 num_warmup=num_warmup, thin_step=thin_step)


def kmc_generator(N, D, target, num_warmup, thin_step, momentum_seed):
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
                            statistics=statistics, num_warmup=num_warmup, thin_step=thin_step)
    job.plot = False
    return job

def kameleon_generator(N, D, target, num_warmup, thin_step):
    start = np.array(start_base + [0. ] * (D - 2))
    
    # estimator parameters
    sigma = 0.46
    nu2 = 4.
    gamma2 = 0.1
    
    if N<100:
        gamma2= 1.
    

    # oracle samples
    Z = sample_banana(N, D, bananicity, V)
    job = KameleonJob(Z, sigma, nu2, gamma2, target, num_iterations, start,
                      statistics=statistics,
                            num_warmup=num_warmup, thin_step=thin_step)
    return job

if __name__ == "__main__":
    logger.setLevel(10)
    Ds = np.sort([2, 8, 16])[::-1]
    Ns = np.sort([10, 50, 100, 200, 500, 1000, 1500, 2000])[::-1]
    
    
    print(Ns)
    print(Ds)
    assert np.min(Ds) >= 2
    num_repetitions = 10
#     num_repetitions = 5
    
    # target
    bananicity = 0.03
    V = 100
    target = Banana(bananicity, V)
    
    # plain MCMC parameters
    num_warmup = 500
    thin_step = 1
    num_iterations = 2000 + num_warmup
#     num_iterations = 2000
#     num_warmup = 0
    
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
    
    aggs_hmc_kmc = {}
    aggs_rw_kameleon = {}
    for N in Ns:
        for D in Ds:
            aggs_hmc_kmc[D] = []
            aggs_hmc_kmc[(N, D)] = []
            aggs_rw_kameleon[D] = []
            aggs_rw_kameleon[(N, D)] = []
            
            
    for i in range(num_repetitions):
        # same momentum for every D and N of every repetition
        momentum_seed += 1
        for D in Ds:
            job = hmc_generator(D, target, num_warmup, thin_step, momentum_seed)
            logger.info("Repetition %d/%d, %s" % (i + 1, num_repetitions, job.get_parameter_fname_suffix()))
            aggs_hmc_kmc[D] += [engine.submit_job(job)]
            job = rw_generator_isotropic(D, target, num_warmup, thin_step)
            logger.info("Repetition %d/%d, %s" % (i + 1, num_repetitions, job.get_parameter_fname_suffix()))
            aggs_rw_kameleon[D] += [engine.submit_job(job)]
            for N in Ns:
                job = kmc_generator(N, D, target, num_warmup, thin_step, momentum_seed)
                logger.info("Repetition %d/%d, %s" % (i + 1, num_repetitions, job.get_parameter_fname_suffix()))
                aggs_hmc_kmc[(N, D)] += [engine.submit_job(job)]
                job = kameleon_generator(N, D, target, num_warmup, thin_step)
                logger.info("Repetition %d/%d, %s" % (i + 1, num_repetitions, job.get_parameter_fname_suffix()))
                aggs_rw_kameleon[(N, D)] += [engine.submit_job(job)]
    
    engine.wait_for_all()
    
    if isinstance(engine, SerialComputationEngine):
        directory = expanduser("~") + os.sep + modulename
        try:
            makedirs(directory)
        except OSError:
            pass
        for D in Ds:
            for agg in aggs_hmc_kmc[D]:
                job_name = unicode(uuid.uuid4())
                agg.store_fire_and_forget_result(directory, job_name)
                
            for N in Ns:
                for agg in aggs_hmc_kmc[(N, D)]:
                    job_name = unicode(uuid.uuid4())
                    agg.store_fire_and_forget_result(directory, job_name)
            
            for agg in aggs_rw_kameleon[D]:
                job_name = unicode(uuid.uuid4())
                agg.store_fire_and_forget_result(directory, job_name)
                
            for N in Ns:
                for agg in aggs_rw_kameleon[(N, D)]:
                    job_name = unicode(uuid.uuid4())
                    agg.store_fire_and_forget_result(directory, job_name)

        # plot some diagnosis
        
        for D in Ds:
            for agg in aggs_hmc_kmc[D]:
                plot_diagnosis(agg, D)
            
            for agg in aggs_rw_kameleon[D]:
                plot_diagnosis(agg, D)
            
            for N in Ns:
                for agg in aggs_hmc_kmc[(N, D)]:
                    plot_diagnosis(agg, D)
                    
                for agg in aggs_rw_kameleon[(N, D)]:
                    plot_diagnosis(agg, D)
