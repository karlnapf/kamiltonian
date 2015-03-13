def log_pdf_estimate(phi_x, theta):
    return theta.dot(phi_x)

def log_pdf_estimate_grad(phi_x_grad, theta):
    return log_pdf_estimate(phi_x_grad, theta)
