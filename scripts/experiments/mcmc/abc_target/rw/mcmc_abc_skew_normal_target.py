from collections import OrderedDict
import os

from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.FileSystem import FileSystem

from kmc.densities.abc_skew_normal import ABCSkewNormalPosterior
from kmc.tools.Log import logger
from kmc.tools.convergence_stats import avg_ess, min_ess
import numpy as np
from scripts.experiments.mcmc.independent_job_classes.MCMCJob import MCMCJobResultAggregatorStoreHome
from scripts.experiments.mcmc.independent_job_classes.RWJob import RWJob
from scripts.experiments.mcmc.independent_job_classes.debug import plot_mcmc_result


modulename = __file__.split(os.sep)[-1].split('.')[-2]
statistics = OrderedDict()
statistics['avg_ess'] = avg_ess
statistics['min_ess'] = min_ess

def rw_generator_isotropic(num_warmup, thin_step):
    # tuned towards roughly 23% acceptance
    sigma_proposal = .4 # marginal sampler
    
    start = np.zeros(10)
    
    target = ABCSkewNormalPosterior()
    job = RWJob(target, num_iterations,
                        start, sigma_proposal,
                        statistics, num_warmup, thin_step)
    
    # marginal sampler
    job.recompute_log_pdf = True
    
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
        job = rw_generator_isotropic(num_warmup, thin_step)
        logger.info("Repetition %d/%d, %s" % (i + 1, num_repetitions, job.get_parameter_fname_suffix()))
        aggs += [engine.submit_job(job)]
        
    engine.wait_for_all()
    
    for i, agg in enumerate(aggs):
        agg.finalize()
        result = agg.get_final_result()
        agg.clean_up()

        if isinstance(engine, SerialComputationEngine):
            plot_mcmc_result(result, D1=0, D2=1)
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
