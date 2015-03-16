from sklearn.cross_validation import KFold

import numpy as np

def sample_basis(D, m, gamma):
    omega = gamma*np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    return omega, u

def feature_map_single(x, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    return np.cos(np.dot(x, omega) + u) * np.sqrt(2. / m)

def feature_map(X, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    
    projection = np.dot(X, omega) + u
    np.cos(projection, projection)
    projection *= np.sqrt(2. / m)
    return projection

def feature_map_derivative_d(X, omega, u, d):
    m = 1 if np.isscalar(u) else len(u)
    
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
    projection *= omega[d, :]
    projection *= np.sqrt(2. / m)
    return -projection

def feature_map_derivative2_d(X, omega, u, d):
    Phi2 = feature_map(X, omega, u)
    Phi2 *= omega[d, :] ** 2
    
    return -Phi2

def feature_map_grad_single(x, omega, u):
    D, m = omega.shape
    grad = np.zeros((m, D))
    
    for d in range(D):
        grad[:, d] = feature_map_derivative_d(x, omega, u, d)
    
    return grad

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

def _objective_sym_completely_manual(X, theta, lmbda, omega, u):
    N = X.shape[0]
    D = X.shape[1]
    m = len(theta)
    
    J_manual = 0.
     
    for n in range(N):
        for d in range(D):
            b_term_manual = -np.sqrt(2. / m) * np.cos(np.dot(X[n], omega) + u) * (omega[d, :] ** 2)
            J_manual -= -np.dot(b_term_manual, theta)
             
            c_vec_manual = -np.sqrt(2. / m) * np.sin(np.dot(X[n], omega) + u) * omega[d, :]
            C_term_manual = np.outer(c_vec_manual, c_vec_manual)
            J_manual += 0.5 * np.dot(theta, np.dot(C_term_manual, theta))
    
    J_manual /= N
    J_manual += 0.5 * lmbda * np.dot(theta, theta)
    return J_manual

def _objective_sym_half_manual(X, theta, lmbda, omega, u):
    N = X.shape[0]
    D = X.shape[1]
    
    J_manual = 0.
     
    for n in range(N):
        for d in range(D):
            b_term = -feature_map_derivative2_d(X[n], omega, u, d)
            J_manual -= np.dot(b_term, theta)

            c_vec = feature_map_derivative_d(X[n], omega, u, d)
            C_term_manual = np.outer(c_vec, c_vec)
            J_manual += 0.5 * np.dot(theta, np.dot(C_term_manual, theta))
    
    J_manual /= N
    J_manual += 0.5 * lmbda * np.dot(theta, theta)
    return J_manual

def compute_b(X, omega, u):
    assert len(X.shape) == 2
    Phi1 = feature_map_derivatives2(X, omega, u)
    return -np.mean(np.sum(Phi1, 0), 0)

def compute_C(X, omega, u):
    assert len(X.shape) == 2
    Phi2 = feature_map_derivatives(X, omega, u)
    d = X.shape[1]
    N = X.shape[0]
    m = Phi2.shape[2]
    C = np.zeros((m, m))
    
    for i in range(N):
        for ell in range(d):
            phi2 = Phi2[ell, i]
            C += np.outer(phi2, phi2)
    
    return C / N

def score_matching_sym(X, lmbda, omega, u, b=None, C=None):
    if b is None:
        b = compute_b(X, omega, u)
        
    if C is None:
        C = compute_C(X, omega, u)
        
    theta = np.linalg.solve(C + lmbda * np.eye(len(C)), b)
    return theta
    
def objective(X, theta, lmbda, omega, u, b=None, C=None):
    if b is None:
        b = compute_b(X, omega, u)
        
    if C is None:
        C = compute_C(X, omega, u)
    
    I = np.eye(len(theta))
    return 0.5 * np.dot(theta, np.dot(C + lmbda * I, theta)) - np.dot(theta, b)


def xvalidate(Z, lmbda, omega, u, n_folds=5, num_repetitions=1):
    Js = np.zeros((num_repetitions, n_folds))
    
    for j in range(num_repetitions):
        kf = KFold(len(Z), n_folds=n_folds, shuffle=True)
        for i, (train, test) in enumerate(kf):
            # train
            theta = score_matching_sym(Z[train], lmbda, omega, u)
            
            # evaluate
            Js[j, i] = objective(Z[test], theta, lmbda, omega, u)
    
    return np.mean(Js, 0)

