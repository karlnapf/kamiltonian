import os
import time

from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.FileSystem import FileSystem

from big_mcmc.tools.Log import logger
from kmc.scripts.experiments.trajectories.plots import plot_trajectory_result_mean_median
import numpy as np


def compute(fname_base, job_generator, Ds, num_repetitions, N, lmbda, num_steps, step_size,
            max_steps=None):
    if not FileSystem.cmd_exists("sbatch"):
        engine = SerialComputationEngine()
        
    else:
        johns_slurm_hack = "#SBATCH --partition=intel-ivy,wrkstn,compute"
        folder = os.sep + os.sep.join(["nfs", "data3", "ucabhst", fname_base])
        batch_parameters = BatchClusterParameters(foldername=folder, max_walltime=24 * 60 * 60,
                                                  resubmit_on_timeout=False, memory=2,
                                                  parameter_prefix=johns_slurm_hack)
        engine = SlurmComputationEngine(batch_parameters, check_interval=1,
                                do_clean_up=True)
    
    aggregators = [[] for _ in range(num_repetitions)]
    for i, D in enumerate(Ds):
        
        # hack that distributes jobs across different queues
        if D > 200:
            engine.batch_parameters.max_walltime = 24 * 60 * 60
            engine.batch_parameters.qos = engine._infer_slurm_qos(engine.batch_parameters.max_walltime,
                                                                  engine.batch_parameters.nodes)
        else:
            engine.batch_parameters.max_walltime = 60 * 60
            
        for j in range(num_repetitions):
            logger.info("%s trajectory, D=%d/%d, repetition %d/%d" % \
                        (str(job_generator), D, Ds.max(), j + 1, num_repetitions))
            job = job_generator(D)
            aggregators[j] += [engine.submit_job(job)]
            time.sleep(0.1)
    
    engine.wait_for_all()
    
    avg_accept = np.zeros((num_repetitions, len(Ds)))
    avg_accept_est = np.zeros((num_repetitions, len(Ds)))
    log_dets = np.zeros((num_repetitions, len(Ds)))
    log_dets_est = np.zeros((num_repetitions, len(Ds)))
    avg_steps_taken = np.zeros((num_repetitions, len(Ds)))
    
    for j in range(len(aggregators)):
        for i in range(len(aggregators[j])):
            agg = aggregators[j][i]
            agg.finalize()
            result = agg.get_final_result()
            agg.clean_up()
            
            avg_accept[j, i] = result.acc_mean
            avg_accept_est[j, i] = result.acc_est_mean
            log_dets[j, i] = result.vol
            log_dets_est[j, i] = result.vol_est
            avg_steps_taken[j, i] = result.steps_taken
            
            
    with open(fname_base + ".npy", 'w+') as f:
        np.savez(f, Ds=Ds, avg_accept=avg_accept, avg_accept_est=avg_accept_est,
                 vols=log_dets, vols_est=log_dets_est, steps_taken=avg_steps_taken)

def process(fname_base, job_generator, Ds, num_repetitions, N, lmbda, num_steps,
            step_size, max_steps):
    fname = fname_base + ".npy"
    # don't recompute if a file exists
    do_compute = False
    if os.path.exists(fname):
        replace = int(raw_input("Replace " + fname + "? "))
    
        if replace:
            do_compute = True
    else:
        do_compute = True
    
    if do_compute:
        compute(fname_base, job_generator, Ds, num_repetitions, N, lmbda, num_steps, step_size, max_steps)
    
    try:
        plot_trajectory_result_mean_median(fname)
#         plot_trajectory_result_boxplot(fname)
#         plot_trajectory_result_boxplot_mix(fname)
    except Exception:
        pass
    
