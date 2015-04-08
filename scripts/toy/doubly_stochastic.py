from kmc.score_matching.random_feats.gaussian_rkhs import sample_basis,\
    feature_map, feature_map_derivatives, feature_map_derivative_d,\
    feature_map_grad_single, feature_map_derivative2_d
from kmc.tools.Log import logger
import numpy as np
from scripts.tools.plotting import evaluate_density_grid, plot_array

import matplotlib.pyplot as plt


N = 10000
D = 2
X = np.random.randn(N, D)

num_iterations = 1000
m = 100
sigma = 1.
gamma = 0.5 / (sigma**2)
learning_rate = lambda t : 1./t
lmbda = 0.01

seed_offset = 0

def predict(x, alphas):
    D = len(x)
    f = 0.
    for i in range(len(alphas)):
        np.random.seed(seed_offset + i)
        omega, u = sample_basis(D=D, m=1, gamma=gamma)
        phi_x = feature_map(x, omega, u)
        f += alphas[i]*phi_x
    
    return f

alphas = np.zeros(num_iterations)
for i in range(num_iterations):
    logger.info("Iteration %d" % i)
    x = X[i]
    
    # sample random feature
    np.random.seed(seed_offset + i)
    omega, u = sample_basis(D=D, m=1, gamma=gamma)
    phi_x = feature_map(x, omega, u)
    
    # sample data point and predict
    f = predict(x, alphas[:i])*phi_x
    
    # gradient of f at x
    f_grad = feature_map_grad_single(x, omega, u) * f
    
    # gradient
    grad = 0
    for d in range(D):
        phi_derivative_d = feature_map_derivative_d(x, omega, u, d)
        phi_derivative2_d = feature_map_derivative2_d(x, omega, u, d)
        
        grad += phi_derivative_d * f_grad[d] + phi_derivative2_d
    
    
    # take gradient step
    r = learning_rate(i+1)
    alphas[i] = -r * grad * phi_x
    
    # down-weight past
    alphas[:i] *= (1-r*lmbda)

# visualise log pdf
log_pdf = lambda x : predict(x, alphas)
res = 20
Xs= np.linspace(-3, 3, res)
Ys = np.linspace(-3, 3, res)

# evaluate density and estimate
G = evaluate_density_grid(Xs, Ys, log_pdf)
plot_array(Xs, Ys, np.exp(G))
plt.show()