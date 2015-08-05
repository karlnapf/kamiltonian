from numpy.linalg.linalg import LinAlgError

import numpy as np


def compute_hamiltonian(Qs, Ps, logq, logp):
    assert len(Ps) == len(Qs)
    return np.asarray([-logq(Qs[i]) - logp(Ps[i]) for i in range(len(Qs))])

def compute_log_accept_pr(q0, p0, Qs, Ps, logq, logp):
    H0 = compute_hamiltonian(q0[np.newaxis, :], p0[np.newaxis, :], logq, logp)
    H = compute_hamiltonian(Qs, Ps, logq, logp)
    
    return np.minimum(np.zeros(H.shape), -H + H0)

def compute_log_accept_pr_single(q0, p0, q, p, logq, logp):
    H0 = compute_hamiltonian(q0[np.newaxis, :], p0[np.newaxis, :], logq, logp)[0]
    H = compute_hamiltonian(q[np.newaxis, :], p[np.newaxis, :], logq, logp)[0]
    return np.minimum(0., -H + H0)

def compute_log_det_trajectory(Qs, Ps):
    joint = np.hstack((Qs, Ps))
    Sigma = np.cov(joint.T) 
    Sigma +=np.eye(Sigma.shape[0])*1e-3
    try:
        L = np.linalg.cholesky(Sigma)
        return 2 * np.sum(np.log(np.diag(L)))
    except LinAlgError:
        return 0.
