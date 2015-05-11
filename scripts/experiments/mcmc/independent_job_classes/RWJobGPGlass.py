from abc import abstractmethod
import os
from os.path import expanduser
import pickle
import uuid

from kameleon_mcmc.gp.GPData import GPData

from kmc.densities.gp_classification_posterior_ard import prior_log_pdf,\
    PseudoMarginalHyperparameters
from kmc.tools.Log import logger
import numpy as np
import scipy as sp
from scripts.experiments.mcmc.independent_job_classes.MCMCJob import MCMCJobResultAggregator
from scripts.experiments.mcmc.independent_job_classes.RWJob import RWJob


class RWJobGPGlass(RWJob):
    def __init__(self, num_iterations, start,
                 sigma_proposal,
                 statistics = {}, num_warmup=500, thin_step=1):
        # target not constructed yet
        RWJob.__init__(self, None, num_iterations, start, sigma_proposal,
               statistics, num_warmup, thin_step)
        
        # store results in home dir straight away
        self.aggregator = MCMCJobResultAggregatorStoreHome()
        
    @abstractmethod
    def set_up(self):
        # load data using kameleon-mcmc code
        logger.info("Loading data")
        X,y=GPData.get_glass_data()
    
        # normalise and whiten dataset, as done in kameleon-mcmc code
        logger.info("Whitening data")
        X-=np.mean(X, 0)
        L=np.linalg.cholesky(np.cov(X.T))
        X=sp.linalg.solve_triangular(L, X.T, lower=True).T
        
        # build target, as in kameleon-mcmc code
        prior = prior_log_pdf
        n_importance = 100
        ridge = 1e-3 
        self.target = PseudoMarginalHyperparameters(X, y, n_importance, prior, ridge, num_shogun_threads=1)


class MCMCJobResultAggregatorStoreHome(MCMCJobResultAggregator):
    @abstractmethod
    def store_fire_and_forget_result(self, folder, job_name):
        home = expanduser("~")
        uni = unicode(uuid.uuid4())
        fname = "%s_ground_truth_iterations=%d_%s.pkl" % (self.__class__.__name__, self.num_iterations, uni)
        full_fname = home + os.sep + fname
        
        with open(full_fname) as f:
            logger.info("Storing result under %s" % full_fname)
            pickle.dump(self, f)
    