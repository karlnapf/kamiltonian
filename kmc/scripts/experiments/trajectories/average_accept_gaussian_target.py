import os
import time

from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.FileSystem import FileSystem

from big_mcmc.tools.Log import logger
from kmc.scripts.experiments.trajectories.independent_jobs_classes.GaussianTrajectoryJob import GaussianTrajectoryJob
from kmc.scripts.experiments.trajectories.plots import plot_trajectory_result
import numpy as np


modulename = __file__.split(os.sep)[-1].split('.')[-2]


def compute(Ds, num_repetitions, N, lmbda, num_steps, step_size):
    if not FileSystem.cmd_exists("sbatch"):
        engine = SerialComputationEngine()
        
    else:
        johns_slurm_hack = "#SBATCH --partition=intel-ivy,wrkstn,compute"
        folder = os.sep + os.sep.join(["nfs", "data3", "ucabhst", modulename])
        batch_parameters = BatchClusterParameters(foldername=folder, max_walltime=24 * 60 * 60,
                                                  resubmit_on_timeout=False, memory=2,
                                                  parameter_prefix=johns_slurm_hack)
        engine = SlurmComputationEngine(batch_parameters, check_interval=1,
                                do_clean_up=True)
    
    aggregators = [[] for _ in range(num_repetitions)]
    for i, D in enumerate(Ds):
        sigma_q = 1.
        sigma_p = 1.
        
        for j in range(num_repetitions):
            logger.info("Gaussian trajectory, D=%d/%d, repetition %d/%d" %\
                        (D, Ds.max(), j+1, num_repetitions))
            job = GaussianTrajectoryJob(N, D, sigma_q, lmbda, sigma_p, num_steps, step_size)
            aggregators[j] += [engine.submit_job(job)]
            time.sleep(0.1)
    
    engine.wait_for_all()
    
    avg_accept = np.zeros((num_repetitions, len(Ds)))
    avg_accept_est = np.zeros((num_repetitions, len(Ds)))
    log_dets = np.zeros((num_repetitions, len(Ds)))
    log_dets_est = np.zeros((num_repetitions, len(Ds)))
    
    for j in range(len(aggregators)):
        for i in range(len(aggregators[j])):
            agg = aggregators[j][i]
            agg.finalize()
            result = agg.get_final_result()
            agg.clean_up()
            
            avg_accept[j, i] = result.acc_mean
            avg_accept_est[j, i] = result.acc_est_mean
            log_dets[j,i] = result.vol
            log_dets_est[j,i] = result.vol_est
            
            
    with open(fname, 'w+') as f:
        np.savez(f, Ds=Ds, avg_accept=avg_accept, avg_accept_est=avg_accept_est,
                 vols=log_dets, vols_est=log_dets_est)

if __name__ == "__main__":
    fname = modulename + ".npy"
    
    # don't recompute if a file exists
    do_compute = False
    if os.path.exists(fname):
        replace = int(raw_input("Replace " + fname + "? "))
    
        if replace:
            do_compute = True
    else:
        do_compute = True
    
    if do_compute:
        Ds = 2 ** np.arange(0, 15)
        num_repetitions = 100
        N = 500
        lmbda = 1.
        num_steps =  1000
        step_size = .1
        compute(Ds, num_repetitions, N, lmbda, num_steps, step_size)
    
    try:
        plot_trajectory_result(fname)
    except Exception:
        pass
    
