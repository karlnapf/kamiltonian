import numpy as np

def SPSA(loss, theta, stepsize, num_repeats=1):
    """
    Implements Simultaneous perturbation stochastic approximation to estimate
    gradient of given loss function
    
    """
    D = len(theta)
    grad_ests = np.zeros((num_repeats, D))
    
    for i in range(num_repeats):
        delta = 2 * (np.random.rand(D) > .5).astype(int) - 1
        thetaplus = theta + stepsize * delta;
        thetaminus = theta - stepsize * delta
        yplus = loss(thetaplus)
        yminus = loss(thetaminus);
        grad_ests[i] = (yplus - yminus) / (2 * stepsize * delta)
    
#     L=np.linalg.cholesky(np.cov(grad_ests.T) + np.eye(D)*1e-5)
#     print 2*np.sum(np.log(np.diag(L)))
    
    grad_mean = np.mean(grad_ests, 0)
    return grad_mean