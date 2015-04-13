import os
import time

from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.FileSystem import FileSystem

from kmc.tools.Log import logger
import numpy as np


def compute(fname_base, job_generator, Ds, Ns, num_repetitions, num_steps, step_size,
            max_steps=None, compute_local=False):
    if not FileSystem.cmd_exists("sbatch") or compute_local:
        engine = SerialComputationEngine()
        
    else:
        johns_slurm_hack = "#SBATCH --partition=intel-ivy,wrkstn,compute"
        folder = os.sep + os.sep.join(["nfs", "data3", "ucabhst", fname_base])
        batch_parameters = BatchClusterParameters(foldername=folder, max_walltime=1 * 60 * 60,
                                                  resubmit_on_timeout=False, memory=3,
                                                  parameter_prefix=johns_slurm_hack,
                                                  nodes=6)
        engine = SlurmComputationEngine(batch_parameters, check_interval=1,
                                do_clean_up=True)
    
    # fixed order of aggregators
    aggregators = []
    for i, D in enumerate(Ds):
        for N in Ns:
            for j in range(num_repetitions):
                logger.info("%s trajectory, D=%d/%d, N=%d/%d repetition %d/%d" % \
                            (str(job_generator), D, np.max(Ds), N, np.max(Ns), j + 1, num_repetitions))
                m = np.min([2000, N])
                job = job_generator(D, N, m)
                aggregators += [engine.submit_job(job)]
                time.sleep(0.1)
    
    engine.wait_for_all()
    
    avg_accept = np.zeros((num_repetitions, len(Ds), len(Ns)))
    avg_accept_est = np.zeros((num_repetitions, len(Ds), len(Ns)))
    log_dets = np.zeros((num_repetitions, len(Ds), len(Ns)))
    log_dets_est = np.zeros((num_repetitions, len(Ds), len(Ns)))
    avg_steps_taken = np.zeros((num_repetitions, len(Ds), len(Ns)))
    
    agg_counter = 0
    for i, D in enumerate(Ds):
        for k,N in enumerate(Ns):
            for j in range(num_repetitions):
                agg = aggregators[agg_counter]
                agg_counter += 1
                agg.finalize()
                result = agg.get_final_result()
                agg.clean_up()
                
                avg_accept[j, i, k] = result.acc_mean
                avg_accept_est[j, i, k] = result.acc_est_mean
                log_dets[j, i, k] = result.vol
                log_dets_est[j, i, k] = result.vol_est
                avg_steps_taken[j, i, k] = result.steps_taken
            
            
    with open(fname_base + ".npy", 'w+') as f:
        np.savez(f, Ds=Ds, Ns=Ns, avg_accept=avg_accept, avg_accept_est=avg_accept_est,
                 vols=log_dets, vols_est=log_dets_est, steps_taken=avg_steps_taken)

def process(fname_base, job_generator, Ds, Ns, num_repetitions, num_steps,
            step_size, max_steps, compute_local=False):
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
        compute(fname_base, job_generator, Ds, Ns, num_repetitions, num_steps, step_size, max_steps, compute_local)
