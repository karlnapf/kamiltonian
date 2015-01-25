def log_pdf_estimate(x, alpha, Z, kernel):
    k = kernel(x, Z)[0, :]
    return alpha.dot(k)

def log_pdf_estimate_grad(x, alpha, Z, kernel_grad):
    k = kernel_grad(x, Z)
    return alpha.dot(k)
