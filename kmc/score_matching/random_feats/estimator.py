import numpy as np

def log_pdf_estimate(phi_x, theta):
    return np.dot(phi_x, theta)

def log_pdf_estimate_grad(phi_x_grad, theta):
    return log_pdf_estimate(phi_x_grad, theta)
