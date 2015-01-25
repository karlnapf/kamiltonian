from kmc.hamiltonian.leapfrog import leapfrog
import matplotlib.pyplot as plt
import numpy as np


def plot_array(Xs, Ys, D):
    """
    Plots a 2D array
    
    Xs - x values the density is evaluated at
    Ys - y values the density is evaluated at
    D - array to plot
    """
    im = plt.imshow(D, origin='lower')
    im.set_extent([Xs.min(), Xs.max(), Ys.min(), Ys.max()])
    im.set_interpolation('nearest')
    im.set_cmap('gray')
    plt.ylim([Ys.min(), Ys.max()])
    plt.xlim([Xs.min(), Xs.max()])
    
def evaluate_density_grid(Xs, Ys, log_pdf, exponentiate=False):
    D = np.zeros((len(Xs), len(Ys)))
    for i in range(len(Xs)):
        for j in range(len(Ys)):
            x = np.array([Xs[i], Ys[j]])
            D[j, i] = log_pdf(x)
    
    return D if exponentiate is False else np.exp(D)

def plot_hamiltonian_trajectory(q, logq, dlogq, logp, dlogp, 
                                p=None, num_steps=50, step_size=.1, plot_H=False):
    
    if p is None:
        p=np.random.randn(len(q))
    
    # run leapfrog  from initial point and momentum
    Qs, Ps = leapfrog(q, dlogq, p, dlogp, step_size, num_steps)
    
    if plot_H:
        plt.subplot(121)
        
    plt.plot(Qs[:, 0], Qs[:, 1], 'r-')
    plt.plot(Qs[0,0], Qs[0, 1], 'r*', markersize=15)
    
    if plot_H:
        plt.subplot(122)
        plt.plot([-logq(Qs[i]) - logp(Ps[i]) for i in range(num_steps)])
    
    return Qs, Ps