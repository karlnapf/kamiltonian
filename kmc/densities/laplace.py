import numpy as np

def log_laplace_pdf(x, scale=1.,compute_grad=False):
    """
    Product of D Laplace distributions positioned at zero. All with the same scale
    """
    D = len(x)
    
    if not compute_grad:
        return -D*np.log(2*scale)-np.sum(np.abs(x))/scale
    else:
        return np.array([(-1. if x[d] >= 0 else 1.) for d in range(D)]) / scale 