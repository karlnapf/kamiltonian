from kmc.score_matching.random_feats.gaussian_rkhs import score_matching_sym, \
    feature_map_single
import matplotlib.pyplot as plt
import numpy as np


# if __name__ == "__main__":
while True:
    sigma = 1.0
    gamma = 0.5/(sigma**2)
    lmbda = 0.0001

    N = 2000
    D = 2
    Z = np.random.randn(N, D)
    m = 200
    omega = gamma * np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    theta = score_matching_sym(Z, lmbda, omega, u)

    # compute log-density and true log density
    Xs = np.linspace(-3, 3)
    Ys = np.linspace(-3, 3)
    D = np.zeros((len(Xs), len(Ys)))
    G = np.zeros(D.shape)
    for i in range(len(Xs)):
        for j in range(len(Ys)):
            x = np.array([Xs[i], Ys[j]])
            phi = feature_map_single(x, omega, u)
            D[j, i] = theta.dot(phi)
            G[j, i] = -np.linalg.norm(x) ** 2
    
    plt.figure(figsize=(16,4))
    plt.subplot(131)
    plt.imshow(G)
    plt.colorbar()
    plt.title("Truth")
    plt.subplot(132)
    plt.imshow(D)
    plt.title("Random features")
    plt.colorbar()
    plt.subplot(133)
    plt.plot(theta)
    plt.title("theta")
    plt.colorbar()
    plt.show()
