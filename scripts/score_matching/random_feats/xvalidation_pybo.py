

"""
Simplest demo performing Bayesian optimization on a one-dimensional test
function. This script also demonstrates user-defined visualization via a
callback function that is imported from the advanced demo.

The `pybo.solve_bayesopt()` function returns a numpy structured array, called
`info` below, which includes the observed input and output data, `info['x']` and
`info['y']`, respectively; and the recommendations made along the way in
`info['xbest']`.

The `callback` function plots the posterior with uncertainty bands, overlaid
onto the true function; below it we plot the acquisition function, and to the
right, the evolution of the recommendation over time.
"""

import os
import sys

import pybo

from kmc.score_matching.random_feats.gaussian_rkhs import xvalidate, \
    sample_basis, score_matching_sym, feature_map_single, \
    feature_map_grad_single
import matplotlib.pyplot as pl
import numpy as np
from scripts.tools.plotting import evaluate_density_grid, plot_array


# import callback from advanced demo
sys.path.append(os.path.dirname(__file__))

def callback_lmbda(model, bounds, info, x, index, ftrue):
    global D
    """
    Plot the current posterior, the index, and the value of the current
    recommendation.
    """
    xmin, xmax = bounds[0]
    xx_ = np.linspace(xmin, xmax, 500)  # define grid
    xx = xx_[:, None]

#     ff = ftrue(xx)                                      # compute true function
    acq = index(xx)  # compute acquisition

    mu, s2 = model.posterior(xx)  # compute posterior and
    lo = mu - 2 * np.sqrt(s2)  # quantiles
    hi = mu + 2 * np.sqrt(s2)

#     ymin, ymax = ff.min(), ff.max()                     # get plotting ranges
#     ymin -= 0.2 * (ymax - ymin)
#     ymax += 0.2 * (ymax - ymin)

    kwplot = {'lw': 2, 'alpha': 0.5}  # common plotting kwargs

    fig = pl.figure(1)
    fig.clf()

    pl.subplot(221)
#     pl.plot(xx, ff, 'k:', **kwplot)                     # plot true function
    pl.plot(xx, mu, 'b-', **kwplot)  # plot the posterior and
    pl.fill_between(xx_, lo, hi, color='b', alpha=0.1)  # uncertainty bands
    pl.scatter(info['x'], info['y'],  # plot data
               marker='o', facecolor='none', zorder=3)
    pl.axvline(x, color='r', **kwplot)  # latest selection
    pl.axvline(info[-1]['xbest'], color='g', **kwplot)  # current recommendation
#     pl.axis((xmin, xmax, ymin, ymax))
    pl.ylabel('posterior')

    pl.subplot(223)
    pl.fill_between(xx_, acq.min(), acq,  # plot acquisition
                    color='r', alpha=0.1)
    pl.axis('tight')
    pl.axvline(x, color='r', **kwplot)  # plot latest selection
    pl.xlabel('input')
    pl.ylabel('acquisition')
    
    pl.subplot(224)
    pl.plot(info['x'], 'g')

    pl.subplot(222)
    lmbda = 2 ** info[-1]['xbest']
    gamma = 0.5*(sigma**2)
    omega, u = sample_basis(D, m, gamma)
    theta = score_matching_sym(Z, lmbda, omega, u)
    logq_est = lambda x: np.dot(theta, feature_map_single(x, omega, u))
    dlogq_est = lambda x: np.dot(theta, feature_map_grad_single(x, omega, u))
    Xs = np.linspace(-3, 3)
    Ys = np.linspace(-3, 3)
    Q = evaluate_density_grid(Xs, Ys, logq_est)
    plot_array(Xs, Ys, Q, pl.gca(), plot_contour=True)
    pl.plot(Z[:, 0], Z[:, 1], 'bx')

    for ax in fig.axes:  # remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    pl.draw()
    pl.show(block=False)

def callback_sigma_lmbda(model, bounds, info, x, index, ftrue):
    global D
    fig = pl.figure(1)
    fig.clf()

    sigma = 2**info[-1]['xbest'][0]
    lmbda = 2 ** info[-1]['xbest'][1]
    pl.title("sigma=%.2f, lmbda=%.6f" % (sigma, lmbda))
    gamma = 0.5*(sigma**2)
    omega, u = sample_basis(D, m, gamma)
    theta = score_matching_sym(Z, lmbda, omega, u)
    logq_est = lambda x: np.dot(theta, feature_map_single(x, omega, u))
    dlogq_est = lambda x: np.dot(theta, feature_map_grad_single(x, omega, u))
    Xs = np.linspace(-3, 3)
    Ys = np.linspace(-3, 3)
    Q = evaluate_density_grid(Xs, Ys, logq_est)
    plot_array(Xs, Ys, Q, pl.gca(), plot_contour=True)
    pl.plot(Z[:, 0], Z[:, 1], 'bx')

    for ax in fig.axes:  # remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    pl.draw()
    pl.show(block=False)

def callback_sigma(model, bounds, info, x, index, ftrue):
    global D
    """
    Plot the current posterior, the index, and the value of the current
    recommendation.
    """
    xmin, xmax = bounds[0]
    xx_ = np.linspace(xmin, xmax, 500)  # define grid
    xx = xx_[:, None]

#     ff = ftrue(xx)                                      # compute true function
    acq = index(xx)  # compute acquisition

    mu, s2 = model.posterior(xx)  # compute posterior and
    lo = mu - 2 * np.sqrt(s2)  # quantiles
    hi = mu + 2 * np.sqrt(s2)

#     ymin, ymax = ff.min(), ff.max()                     # get plotting ranges
#     ymin -= 0.2 * (ymax - ymin)
#     ymax += 0.2 * (ymax - ymin)

    kwplot = {'lw': 2, 'alpha': 0.5}  # common plotting kwargs

    fig = pl.figure(1)
    fig.clf()

    pl.subplot(221)
#     pl.plot(xx, ff, 'k:', **kwplot)                     # plot true function
    pl.plot(xx, mu, 'b-', **kwplot)  # plot the posterior and
    pl.fill_between(xx_, lo, hi, color='b', alpha=0.1)  # uncertainty bands
    pl.scatter(info['x'], info['y'],  # plot data
               marker='o', facecolor='none', zorder=3)
    pl.axvline(x, color='r', **kwplot)  # latest selection
    pl.axvline(info[-1]['xbest'], color='g', **kwplot)  # current recommendation
#     pl.axis((xmin, xmax, ymin, ymax))
    pl.ylabel('posterior')

    pl.subplot(223)
    pl.fill_between(xx_, acq.min(), acq,  # plot acquisition
                    color='r', alpha=0.1)
    pl.axis('tight')
    pl.axvline(x, color='r', **kwplot)  # plot latest selection
    pl.xlabel('input')
    pl.ylabel('acquisition')
    
    pl.subplot(224)
    pl.plot(info['x'], 'g')

    pl.subplot(222)
    gamma = 2 ** info[-1]['xbest']
    omega, u = sample_basis(D, m, gamma)
    theta = score_matching_sym(Z, lmbda, omega, u)
    logq_est = lambda x: np.dot(theta, feature_map_single(x, omega, u))
    dlogq_est = lambda x: np.dot(theta, feature_map_grad_single(x, omega, u))
    Xs = np.linspace(-3, 3)
    Ys = np.linspace(-3, 3)
    Q = evaluate_density_grid(Xs, Ys, logq_est)
    plot_array(Xs, Ys, Q, pl.gca(), plot_contour=True)
    pl.plot(Z[:, 0], Z[:, 1], 'bx')

    for ax in fig.axes:  # remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    pl.draw()
    pl.show(block=False)

def generator_sigma_objective():
    def _f(log2_sigma):
        sigma = 2 ** log2_sigma
        folds = np.zeros(num_repetitions)
        for i in range(num_repetitions):
            gamma = 0.5 * (sigma ** 2)
            omega, u = sample_basis(D, m, gamma)
            folds[i] = np.mean(xvalidate(Z, lmbda, omega, u, num_folds, num_repetitions=1))
            
            
        result = np.mean(folds)
        print("xvalidation, sigma: %.2f, lambda: %.6f, J=%.3f" % \
                    (sigma, lmbda, result))
        
        # transform to log space to make GP work.
        # since unbounded, add constant
        # also pybo maximises so invert sign
        result = -np.log(result + 100)
        
        
        return result
    
    return _f

def generator_lmbda_objective():
    def _f(log2_lmbda):
        lmbda = 2 ** log2_lmbda
        gamma = 0.5 * (sigma ** 2)
        
        folds = np.zeros(num_repetitions)
        for i in range(num_repetitions):
            omega, u = sample_basis(D, m, gamma)
            folds[i] = np.mean(xvalidate(Z, lmbda, omega, u, num_folds, num_repetitions=1))
            
            
        result = np.mean(folds)
        print("xvalidation, sigma: %.2f, lambda: %.6f, J=%.3f" % \
                    (sigma, lmbda, result))
        
        # transform to log space to make GP work.
        # since unbounded, add constant
        # also pybo maximises so invert sign
        result = -np.log(result + 100)
        
        
        return result
    
    return _f

def generator_sigma_lmbda_objective():
    def _f(log2_sigma_lmbda):
        sigma = 2** log2_sigma_lmbda[0]
        lmbda = 2 ** log2_sigma_lmbda[1]
        gamma = 0.5 * (sigma ** 2)
        
        folds = np.zeros(num_repetitions)
        for i in range(num_repetitions):
            omega, u = sample_basis(D, m, gamma)
            folds[i] = np.mean(xvalidate(Z, lmbda, omega, u, num_folds, num_repetitions=1))
            
            
        result = np.mean(folds)
        print("xvalidation, sigma: %.2f, lambda: %.6f, J=%.3f" % \
                    (sigma, lmbda, result))
        
        # transform to log space to make GP work.
        # since unbounded, add constant
        # also pybo maximises so invert sign
        result = -np.log(result + 100)
        
        
        return result
    
    return _f

if __name__ == '__main__':
    N = 200
    D = 2
    Z = np.random.randn(N, D)
    m = N
    num_folds = 5
    num_repetitions = 10
    lmbda = 0.0001
    sigma = 0.92

    objective = generator_sigma_lmbda_objective()
    
    info = pybo.solve_bayesopt(
        objective,
        bounds=np.array([[-5,5],[-20, 1]]),
        noisefree=False,
        callback=callback_sigma_lmbda,
        niter=15)
     
    print("x")
    print(info['x'])
     
    print("y")
    print(info['y'])
     
    print("xbest")
    print(info['xbest'])
     
    pl.figure()
    pl.plot(info['x'], info['y'], 'o')
    pl.show()
    
