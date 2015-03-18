import numpy as np


def log_sum_exp(X):
    """
    Computes log sum_i exp(X_i).
    Useful if you want to solve log \int f(x)p(x) dx
    where you have samples from p(x) and can compute log f(x)
    """
    # extract minimum
    X0 = X.min()
    X_without_X0 = np.delete(X, X.argmin())
    
    return X0 + np.log(1 + np.sum(np.exp(X_without_X0 - X0)))

def log_mean_exp(X):
    """
    Computes log 1/n sum_i exp(X_i).
    Useful if you want to solve log \int f(x)p(x) dx
    where you have samples from p(x) and can compute log f(x)
    """
    
    return log_sum_exp(X) - np.log(len(X))

def avg_prob_of_log_probs(X):
    """
    Given a set of log-probabilities, this computes log-mean-exp of them.
    Careful checking is done to prevent buffer overflows
    Similar to calling (but overflow-safe): log_mean_exp(log_prob)
    """
    
    # extract inf inds (no need to delete X0 from X here)
    X0 = X.min()
    inf_inds = np.isinf(np.exp(X - X0))
    
    # remove these numbers
    X_without_inf = X[~inf_inds]
    
    # return exp-log-mean-exp on shortened array
    avg_prob_without_inf = np.exp(log_mean_exp(X_without_inf))
    
    # re-normalise by the full length, which is equivalent to adding a zero probability observation
    renormaliser = float(len(X_without_inf)) / len(X)
    avg_prob_without_inf = avg_prob_without_inf * renormaliser
    
    return avg_prob_without_inf
