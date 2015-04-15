import numpy as np


def log_student_pdf(x, nu=1., compute_grad=False):
    """
    Product of D Student's T distributions. All with the same degree of freedom
    """
    if compute_grad:
        return -(nu + 1.) / nu * x
    else:
        return -(nu + 1.) / 2. * np.sum(np.log(1. + (x ** 2.) / nu))
