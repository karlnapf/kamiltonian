from rpy2 import robjects
import rpy2
from rpy2.robjects.numpy2ri import ri2numpy

from kmc.densities.abc import ABCPosterior
from kmc.densities.gaussian import log_gaussian_pdf
from kmc.tools.Log import logger
import numpy as np


robjects.r('library(sn)')
rpy2.robjects.numpy2ri.activate()


def sample_skew_normal(N, mu=np.zeros(2), Sigma=np.eye(2), alphas = np.ones(2)):
    D = len(mu)
    assert len(mu.shape) == 1
    assert len(alphas.shape) == 1
    assert len(Sigma.shape) == 2
    
    assert D == Sigma.shape[0]
    assert D == Sigma.shape[1]
    assert D == len(alphas)
    
#     rdata = sn.rmsn(N, mu, Sigma, alphas)
    
    
    r_rmsn = robjects.r['rmsn']
    r_alpha = robjects.FloatVector(alphas)
    r_xi = robjects.FloatVector(mu)
    r_Sigma_vec = robjects.FloatVector(Sigma.transpose().reshape(Sigma.size))
    r_Omega = robjects.r.matrix(r_Sigma_vec, nrow=D, ncol=D)
    
    result = r_rmsn(N, r_xi, r_Omega, r_alpha)
    return ri2numpy(result)

def wide_zero_mean_normal_prior(x):
    D = len(x)
    mu = np.zeros(D)
    Sigma = np.eye(D) * 5
    return log_gaussian_pdf(x, mu, Sigma, is_cholesky=True)

def skew_normal_simulator(theta):
    D = len(theta)
    Sigma = np.eye(D)
    alphas = np.zeros(D) + 10
    N = 10
    
    return sample_skew_normal(N, theta, Sigma, alphas)

class ABCSkewNormalPosterior(ABCPosterior):
    def __init__(self, D=10, n_lik_samples=10, epsilon=14., prior=wide_zero_mean_normal_prior):
        
        ABCPosterior.__init__(self, skew_normal_simulator, n_lik_samples, epsilon, prior)
        self.D = D
        
    def set_up(self):
        logger.info("Generating dataset")
        # true vale of theta
        theta = np.zeros(self.D)
        
        old = np.random.get_state()
        np.random.seed(0)
        self.data = skew_normal_simulator(theta)
        np.random.set_state(old)
    

if __name__ == "__main__":
    N=1000
    D=2
    mu = np.zeros(D)
    Sigma = np.eye(D)
    alphas = np.zeros(D) + 10
    samples = sample_skew_normal(N, mu, Sigma, alphas)
    
    import seaborn as sns
    sns.kdeplot(samples)
    
    m = np.mean(samples, 0)
    sns.plt.plot([m[0]], [m[1]], 'x', markersize=15)
    
    sns.plt.show()
    