import numpy as np
from kmc.score_matching.random_feats.gaussian_rkhs import feature_map_single,\
    feature_map_grad_single

def log_pdf_estimate(phi_x, theta):
    return np.dot(phi_x, theta)

def log_pdf_estimate_grad(phi_x_grad, theta):
    return log_pdf_estimate(phi_x_grad, theta)

class RandomFeatsEstimator(object):
    def __init__(self, theta, omega, u):
        self.theta = theta
        self.omega = omega
        self.u = u
    
    def log_pdf(self, x):
        phi = feature_map_single(x, self.omega, self.u)
        return np.dot(phi, self.theta)
    
    def grad(self, x):
        phi_grad = feature_map_grad_single(x, self.omega, self.u)
        return np.dot(phi_grad, self.theta)