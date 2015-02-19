import numpy as np


def compute_hamiltonian(Qs, Ps, logq, logp):
    assert len(Ps) == len(Qs)
    return np.asarray([-logq(Qs[i]) - logp(Ps[i]) for i in range(len(Qs))])

def compute_log_accept_pr(q0, p0, Qs, Ps, logq, logp):
    H0 = compute_hamiltonian(q0[np.newaxis, :], p0[np.newaxis, :], logq, logp)
    H = compute_hamiltonian(Qs, Ps, logq, logp)
    
    return np.minimum(np.zeros(H.shape), H - H0)

def compute_log_det_trajectory(Qs, Ps):
    joint = np.hstack((Qs, Ps))
    Sigma = np.cov(joint.T)
    L = np.linalg.cholesky(Sigma)
    return 2 * np.sum(np.log(np.diag(L)))
