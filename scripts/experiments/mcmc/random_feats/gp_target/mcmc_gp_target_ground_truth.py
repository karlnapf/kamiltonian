from collections import OrderedDict
import os
import pickle
import uuid

from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.FileSystem import FileSystem

from kmc.tools.Log import logger
from kmc.tools.convergence_stats import avg_ess
import numpy as np
from scripts.experiments.mcmc.independent_job_classes.RWJobGPGlass import RWJobGPGlass
from scripts.experiments.mcmc.independent_job_classes.debug import plot_diagnosis_single_instance


modulename = __file__.split(os.sep)[-1].split('.')[-2]
statistics = OrderedDict()
statistics['avg_ess'] = avg_ess

def rw_generator_isotropic(num_warmup, thin_step):
    # tuned towards roughly 23% acceptance
    sigma_proposal = 0.54
    
    start = np.random.randn(9) * 10
    
    return RWJobGPGlass(num_iterations,
                        start, sigma_proposal,
                        statistics, num_warmup, thin_step)

if __name__ == "__main__":
    logger.setLevel(10)
    num_repetitions = 50
    
    # plain MCMC parameters, plan is to use every 200th sample
    thin_step = 1
    num_iterations = 20000
    num_warmup = 1000
    
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
    
    aggs_rw = []
    
    for i in range(num_repetitions):
        job = rw_generator_isotropic(num_warmup, thin_step)
        logger.info("Repetition %d/%d, %s" % (i + 1, num_repetitions, job.get_parameter_fname_suffix()))
        aggs_rw += [engine.submit_job(job)]
        
    engine.wait_for_all()
    
    for i, agg in enumerate(aggs_rw):
        if isinstance(engine, SerialComputationEngine):
            plot_diagnosis_single_instance(agg, D1=1, D2=6)
            
        agg.finalize()
        mcmc_job = agg.get_final_result().mcmc_job
        agg.clean_up()
        
        # print some summary stats
        accepted = mcmc_job.accepted
        avg_ess = mcmc_job.posterior_statistics["avg_ess"]
        
        time = mcmc_job.time_taken_sampling
        time_set_up = mcmc_job.time_taken_set_up
    
        logger.info("Repetition %d" % i)
        logger.info("Average acceptance probability: %.2f" % np.mean(accepted))
        logger.info("Average ESS: %.2f" % avg_ess)
        logger.info("Total time: %.2f" % (time + time_set_up))
        
        
        # save result under unique filename
        fname = "%s_ground_truth_iterations=%d_%s.pkl" % (modulename, num_iterations, unicode(uuid.uuid4()))
        with open(fname, 'w+') as f:
            logger.info("Storing result under %s" % fname)
            pickle.dump(mcmc_job, f)
