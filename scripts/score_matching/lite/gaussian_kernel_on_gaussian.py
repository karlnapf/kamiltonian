from kmc.score_matching.kernel.incomplete_cholesky import incomplete_cholesky_gaussian
from kmc.score_matching.kernel.kernels import gaussian_kernel
from kmc.score_matching.lite.gaussian_rkhs import score_matching_sym,\
    score_matching_sym_low_rank
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    sigma = 1.
    lmbda = 1.
    N = 1000
    Z = np.random.randn(N, 2)
    
    a = score_matching_sym(Z, sigma, lmbda)
    R = incomplete_cholesky_gaussian(Z, sigma, eta=0.1)["R"]
    print("Low rank dimension: %d/%d" % (R.shape[0], N))
    a_chol = score_matching_sym_low_rank(Z, sigma, lmbda, L=R.T)
    

    # compute log-density and true log density
    Xs = np.linspace(-4, 4)
    Ys = np.linspace(-4, 4)
    D = np.zeros((len(Xs), len(Ys)))
    D_chol = np.zeros(D.shape)
    G = np.zeros(D.shape)
    for i in range(len(Xs)):
        for j in range(len(Ys)):
            x = np.array([[Xs[i], Ys[j]]])
            k = gaussian_kernel(Z, x)[:, 0]
            D[j, i] = k.dot(a)
            D_chol[j, i] = k.dot(a_chol)
            G[j, i] = -np.linalg.norm(x) ** 2
    
    plt.subplot(131)
    plt.imshow(G)
    plt.title("Truth")
    plt.subplot(132)
    plt.imshow(D)
    plt.title("Lite")
    plt.subplot(133)
    plt.imshow(D_chol)
    plt.title("Lite-Cholesky")
    plt.show()
