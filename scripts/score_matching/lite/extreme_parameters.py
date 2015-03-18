
from kmc.densities.gaussian import sample_gaussian
from kmc.score_matching.kernel.kernels import gaussian_kernel
from kmc.score_matching.lite.gaussian_rkhs import score_matching_sym,\
    _compute_b_sym, _compute_C_sym, _objective_sym
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    np.random.seed(0)
    D = 2
    
    # true target log density
    Sigma = np.diag(np.linspace(0.01, 1, D))
    Sigma[:2, :2] = np.array([[1, .95], [.95, 1]])
    L = np.linalg.cholesky(Sigma)

    # estimate density in rkhs
    N = 200
    Z = sample_gaussian(N, Sigma=L, is_cholesky=True)
    sigma = 3.
    lmbda = .1
    
    K = gaussian_kernel(Z, sigma=sigma)
    
    a = score_matching_sym(Z, sigma, lmbda, K)

    b = _compute_b_sym(Z, K, sigma)
    C = _compute_C_sym(Z, K, sigma)
    J = _objective_sym(Z, sigma, lmbda, a, K, b, C)
    
    plt.figure()
    plt.plot(b)
    plt.title("b")
    
    plt.figure()
    plt.imshow(C)
    plt.title("C")
    plt.colorbar()
    
    plt.figure()
    plt.imshow(K)
    plt.title("K")
    plt.colorbar()
    
    plt.figure()
    plt.plot(a)
    plt.title("a")
    
    plt.figure()
    plt.plot(a*b)
    plt.title("a*b")
    
    plt.figure()
    plt.plot(np.sum(K, 1))
    plt.title("K 1")
    
    print("Objective:", J)
    print("a^T b", a.dot(b))
    print("a^T C a", a.dot(C.dot(a)))
    

    plt.show()