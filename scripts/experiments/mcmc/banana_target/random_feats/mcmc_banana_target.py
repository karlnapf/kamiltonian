from collections import OrderedDict
from os import makedirs
import os
from os.path import expanduser
from scipy.spatial.distance import squareform, pdist
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
from kmc.tools.convergence_stats import avg_ess, min_ess
import numpy as np
from scripts.experiments.mcmc.independent_job_classes.HMCJob import HMCJob
from scripts.experiments.mcmc.independent_job_classes.KMCRandomFeatsJob import KMCRandomFeatsJob
from scripts.experiments.mcmc.independent_job_classes.KameleonJob import KameleonJob
from scripts.experiments.mcmc.independent_job_classes.RWJob import RWJob
from scripts.experiments.mcmc.independent_job_classes.debug import plot_diagnosis


modulename = __file__.split(os.sep)[-1].split('.')[-2]
start_base = [0, -3.]

statistics = OrderedDict()
statistics['avg_quantile_error']=kmc.densities.banana.avg_quantile_error
statistics['avg_ess']=avg_ess
statistics['min_ess']=min_ess
statistics['norm_of_mean']=norm_of_emp_mean

def get_start(D):
    start = np.array(start_base + [0. ] * (D - 2))
    start = np.random.randn(D)*10
    start = np.ones(D) * 10
    return start

def hmc_generator(D, target, num_warmup, thin_step, momentum_seed):
    # determined by pilot runs
    if D == 2:
        step_size_min = 0.8
        step_size_max = 1.5
    elif D==8:
        step_size_min = 0.6
        step_size_max = 1.3
    
    start = get_start(D)
    momentum = IsotropicZeroMeanGaussian(sigma=sigma_p, D=D)
    return HMCJob(target, momentum, num_iterations, start,
                  num_steps_min, num_steps_max, step_size_min, step_size_max, momentum_seed,
                  statistics=statistics, num_warmup=num_warmup, thin_step=thin_step)

def rw_generator_isotropic(D, target, num_warmup, thin_step):
    # tuned towards roughly 23% acceptance
    sigmas_proposal = {
              2: 4.4,
              8: 1.,
              16: 0.35,
              32: 0.05
              }
    start = get_start(D)
    return RWJob(target, num_iterations, start, sigmas_proposal[D],
                 statistics=statistics,
                 num_warmup=num_warmup, thin_step=thin_step)


def kmc_generator(N, D, target, num_warmup, thin_step, momentum_seed):
    if D == 2:
        step_size_min = 0.8
        step_size_max = 1.5
    elif D==8:
        step_size_min = 0.6
        step_size_max = 1.3
    
    start = get_start(D)
    momentum = IsotropicZeroMeanGaussian(sigma=sigma_p, D=D)
    
    # estimator parameters
    sigma = 0.46
    lmbda = 0.000001

    learn_parameters = True if N < 500 else False
    force_relearn_parameters = True if N < 500 else False

    # oracle samples
    Z = sample_banana(N, D, bananicity, V)
    job = KMCRandomFeatsJob(Z, N, sigma, lmbda,
                            target, momentum, num_iterations,
                            start, num_steps_min, num_steps_max,
                            step_size_min, step_size_max, momentum_seed, learn_parameters=learn_parameters,
                            force_relearn_parameters=force_relearn_parameters,
                            statistics=statistics, num_warmup=num_warmup, thin_step=thin_step)
    job.plot = False
    return job

def kameleon_generator(N, D, target, num_warmup, thin_step):
    # determined by pilot runs
    if N==2000 and D==8:
        nu2 = .5
        gamma2 = .1
    elif N==1500 and D==8:
        nu2 = .7
        gamma2 = .1
    elif N==1000 and D==8:
        nu2 = .8
        gamma2 = .1
    elif N==500 and D==8:
        nu2 = .8
        gamma2 = .5
    elif N==200 and D==8:
        nu2 = 1.
        gamma2 = 1.
    elif N==100 and D==8:
        nu2 = 2.
        gamma2 = 1.
    elif N==50 and D==8:
        nu2 = 2.
        gamma2 = 1.
    # oracle samples
    Z = sample_banana(N, D, bananicity, V)
    
    # median heuristic:
    dists=squareform(pdist(Z, 'sqeuclidean'))
    median_dist=np.median(dists[dists>0])
    sigma=0.5*median_dist
    
    start = get_start(D)
    job = KameleonJob(Z, sigma, nu2, gamma2, target, num_iterations, start,
                      statistics=statistics,
                            num_warmup=num_warmup, thin_step=thin_step)
    return job

if __name__ == "__main__":
    logger.setLevel(10)
    Ds = np.sort([8])[::-1]
    Ns = np.sort([50, 100, 200, 500, 1000, 1500, 2000])[::-1]
    Ns = np.sort([2000])[::-1]
    
    print(Ns)
    print(Ds)
    assert np.min(Ds) >= 2
    num_repetitions = 10
    num_repetitions = 1
    
    # target
    bananicity = 0.03
    V = 100
    target = Banana(bananicity, V)
    
    # plain MCMC parameters
    num_warmup = 500
    thin_step = 1
    num_iterations = 2000 + num_warmup
    num_iterations = 100
    num_warmup = 0
    
    # hmc parameters
    num_steps_min = 10
    num_steps_max = 100
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
                print(agg.__class__.__name__)
                plot_diagnosis(agg, D)
            
            for agg in aggs_rw_kameleon[D]:
                print(agg.__class__.__name__)
                plot_diagnosis(agg, D)
            
            for N in Ns:
                for agg in aggs_hmc_kmc[(N, D)]:
                    print(agg.__class__.__name__)
                    plot_diagnosis(agg, D)
                    
                for agg in aggs_rw_kameleon[(N, D)]:
                    print(agg.__class__.__name__)
                    plot_diagnosis(agg, D)
