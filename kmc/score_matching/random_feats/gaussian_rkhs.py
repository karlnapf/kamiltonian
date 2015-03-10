import numpy as np


def feature_map_single(x, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    return np.cos(np.dot(x, omega) + u) * np.sqrt(2.) / np.sqrt(m)

def feature_map(X, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    
    projection = np.dot(X, omega) + u
    np.cos(projection, projection)
    projection *= np.sqrt(2.) / np.sqrt(m)
    return projection

def feature_map_derivative_d(X, omega, u, d):
    m = 1 if np.isscalar(u) else len(u)
    
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
    projection *= omega[d, :]
    projection *= np.sqrt(2.) / np.sqrt(m)
    return -projection

def feature_map_derivative2_d(X, omega, u, d):
    Phi2 = feature_map(X, omega, u)
    Phi2 *= omega[d, :] ** 2
    
    return Phi2
    
def feature_map_derivatives_loop(X, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    projections = np.zeros((D, N, m))
    for d in range(D):
        projections[d, :, :] = feature_map_derivative_d(X, omega, u, d)
        
    return projections

def feature_map_derivatives2_loop(X, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    projections = np.zeros((D, N, m))
    for d in range(D):
        projections[d, :, :] = feature_map_derivative2_d(X, omega, u, d)
        
    return projections

def feature_map_derivatives(X, omega, u):
    return feature_map_derivatives_loop(X, omega, u)

def feature_map_derivatives2(X, omega, u):
    return feature_map_derivatives2_loop(X, omega, u)

def compute_b(X, omega, u):
    Phi1 = feature_map_derivatives(X, omega, u)
    return np.mean(np.sum(Phi1, 0), 0)

def compute_C(X, omega, u):
    Phi2 = feature_map_derivatives2(X, omega, u)
    d = X.shape[1]
    n = X.shape[0]
    m = Phi2.shape[2]
    C = np.zeros((m, m))
    
    for i in range(n):
        for ell in range(d):
            phi2 = Phi2[ell, i]
            C += np.outer(phi2, phi2)
    
    return C / n
