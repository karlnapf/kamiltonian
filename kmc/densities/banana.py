from kameleon_mcmc.distribution.Banana import Banana
from kameleon_mcmc.tools.Visualise import Visualise
from theano import function
import theano

import numpy as np
import theano.tensor as T


def log_banana_pdf_theano_expr(x, bananicity, V):
    transformed = x.copy()
    transformed = T.set_subtensor(transformed[1], x[1] - bananicity * ((x[0] ** 2) - V))
    transformed = T.set_subtensor(transformed[0], x[0] / T.sqrt(V))
    
    log_determinant_part = 0.
    quadratic_part = -0.5 * transformed.dot(transformed)
    const_part = -0.5 * x.shape[0] * np.log(2 * np.pi)
    
    banana_log_pdf_expr = const_part + log_determinant_part + quadratic_part
    return banana_log_pdf_expr

# build theano functions for log-pdf and gradient
x = T.dvector('x')
bananicity = T.dscalar('bananicity')
V = T.dscalar('V')
banana_log_pdf_expr = log_banana_pdf_theano_expr(x, bananicity, V)
banana_log_pdf_theano = function([x, bananicity, V], banana_log_pdf_expr)
banana_log_pdf_grad_theano = function([x, bananicity, V], theano.gradient.jacobian(banana_log_pdf_expr, x))

def log_banana_pdf(x, bananicity=0.03, V=100, compute_grad=False):
    if not compute_grad:
        return banana_log_pdf_theano(x, bananicity, V)
    else:
        return banana_log_pdf_grad_theano(x, bananicity, V)

def sample_banana(N, D, bananicity=0.03, V=100):
    return  Banana(D, bananicity, V).sample(N).samples

if __name__ == "__main__":
    Xs = np.linspace(-20, 20)
    Ys = np.linspace(-10, 10)
    D = np.zeros((len(Xs), len(Ys)))
    # compute log-density
    for i in range(len(Xs)):
        for j in range(len(Ys)):
            x = np.array([Xs[i], Ys[j]])
            D[j, i] = np.linalg.norm(log_banana_pdf(x, compute_grad=True))
    Visualise.plot_array(Xs, Ys, D)
    import matplotlib.pyplot as plt
    plt.show()