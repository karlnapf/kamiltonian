import numpy as np

def log_sum_exp(X):
    """
    Computes log sum_i exp(X_i).
    Useful if you want to solve log \int f(x)p(x) dx
    where you have samples from p(x) and can compute log f(x)
    """
    # extract minimum
    X0=X.min()
    X_without_X0=np.delete(X,X.argmin())
    
    return X0+np.log(1+np.sum(np.exp(X_without_X0-X0)))

def log_mean_exp(X):
    """
    Computes log 1/n sum_i exp(X_i).
    Useful if you want to solve log \int f(x)p(x) dx
    where you have samples from p(x) and can compute log f(x)
    """
    
    return log_sum_exp(X)-np.log(len(X))