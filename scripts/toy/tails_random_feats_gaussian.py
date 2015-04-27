from prettytable import PrettyTable

from kmc.densities.gaussian import sample_gaussian, log_gaussian_pdf, \
    emp_quantiles
from kmc.hamiltonian.hamiltonian import compute_log_accept_pr_single, \
    compute_log_accept_pr
from kmc.hamiltonian.leapfrog import leapfrog
from kmc.score_matching.random_feats.estimator import log_pdf_estimate_grad, \
    log_pdf_estimate
from kmc.score_matching.random_feats.gaussian_rkhs import sample_basis, \
    score_matching_sym, feature_map_grad_single, feature_map_single, objective
from kmc.tools.convergence_stats import autocorr
import matplotlib.pyplot as plt
import numpy as np
from scripts.tools.plotting import evaluate_density_grid, plot_array, \
    plot_2d_trajectory, evaluate_gradient_grid


# target
D = 2
bananicity = 0.03
L = np.linalg.cholesky(np.eye(D))
dlogq = lambda x: log_gaussian_pdf(x, Sigma=L, is_cholesky=True, compute_grad=True)
logq = lambda x: log_gaussian_pdf(x, Sigma=L, is_cholesky=True, compute_grad=False)

# oracle samples
N = 500
Z = sample_gaussian(N, mu=np.zeros(D), Sigma=L, is_cholesky=True)

# fit density in RKHS from oracle samples
sigma = 0.5
gamma = 0.5 * (sigma ** 2)
lmbda = 0.0008
m = N
gamma = 0.5 * (sigma ** 2)
omega, u = sample_basis(D, m, gamma)
theta = score_matching_sym(Z, lmbda, omega, u)
logq_est = lambda x: log_pdf_estimate(feature_map_single(x, omega, u), theta)
dlogq_est = lambda x: log_pdf_estimate_grad(feature_map_grad_single(x, omega, u),
                                            theta)

print("J=%.2f" % objective(Z, theta, lmbda, omega, u))

# plot density estimate
plt.figure(figsize=(8, 4))
Xs = np.linspace(-5, 5)
Ys = np.linspace(-5, 5)
Xs_grad = np.linspace(-25, 25, 40)
Ys_grad = np.linspace(-25, 25, 40)
G = evaluate_density_grid(Xs, Ys, logq_est)
G_norm, U, V, _, _ = evaluate_gradient_grid(Xs_grad, Ys_grad, dlogq_est)
plt.subplot(121)
plt.plot(Z[:, 0], Z[:, 1], 'bx')
plot_array(Xs, Ys, np.exp(G), plot_contour=True)
plt.subplot(122)
plot_array(Xs_grad, Ys_grad, G_norm, plot_contour=True)
plt.quiver(Xs_grad, Ys_grad, U, V, color='m')
plt.show()

# plain MCMC parameters
thin = 20
num_warmup = 10
num_iterations = 1000 + num_warmup * thin
q_current = np.zeros(D) + 20
q_current_est = q_current

# hmc parameters
num_steps_min = 10
num_steps_max = 100
step_size_min = 0.05
step_size_max = 0.3
sigma_p = 1.
L_p = np.linalg.cholesky(np.eye(D) * sigma_p)
p_sample = lambda: sample_gaussian(N=1, mu=np.zeros(D), Sigma=L_p, is_cholesky=True)[0]
logp = lambda x: log_gaussian_pdf(x, mu=np.zeros(D), Sigma=L_p, compute_grad=False, is_cholesky=True)
dlogp = lambda x: log_gaussian_pdf(x, mu=np.zeros(D), Sigma=L_p, compute_grad=True, is_cholesky=True)

# MCMC results
samples = np.zeros((num_iterations, D))
accepted = np.zeros(num_iterations)
acc_prob = np.zeros(num_iterations)
proposals = np.zeros(samples.shape)
momentums = np.zeros(samples.shape)
log_pdf = np.zeros(num_iterations)

samples_est = np.zeros((num_iterations, D))
proposals_est = np.zeros(samples.shape)
acc_prob_est = np.zeros(num_iterations)
accepted_est = np.zeros(num_iterations)
log_pdf_est = np.zeros(num_iterations)

# run MCMC
for i in range(num_iterations):
    print("%d/%d" % (i + 1, num_iterations))
    # sample momentum
    p = p_sample()
    momentums[i] = p
    
    # simulate Hamiltonian flow, use last point as proposal
    num_steps = np.random.randint(num_steps_min, num_steps_max + 1)
    step_size = np.random.rand() * (step_size_max - step_size_min) + step_size_min
    Qs, Ps = leapfrog(q_current, dlogq, p, dlogp, step_size, num_steps)
    Qs_est, Ps_est = leapfrog(q_current_est, dlogq_est, p, dlogp, step_size, num_steps)
    proposals[i] = Qs[-1]
    proposals_est[i] = Qs_est[-1]
    
    # compute acceptance probability
    acc_prob[i] = np.exp(compute_log_accept_pr_single(q_current, p, Qs[-1], Ps[-1], logq, logp))
    acc_prob_est[i] = np.exp(compute_log_accept_pr_single(q_current_est, p, Qs_est[-1], Ps_est[-1], logq, logp))
    
    # accept-reject 
    r = np.random.rand()
    accepted[i] = r < acc_prob[i]
    accepted_est[i] = r < acc_prob_est[i]

    if False and True:
        # visualise trajectories and acceptance probability along
        plt.figure(figsize=(10,10))
        plt.subplot(121)
        plot_array(Xs, Ys, np.exp(G), plot_contour=False)
        plot_2d_trajectory(Qs, "r-")
        plot_2d_trajectory(Qs_est, "b-")
        plt.title("HMC: %.2f, accepted: %d\nKMC:  %.2f, accepted: %d" % (acc_prob[i], accepted[i],
                                                    acc_prob_est[i], accepted_est[i]))
        plt.subplot(122)
        plot_array(Xs_grad, Ys_grad, G_norm, plot_contour=False)
        plot_2d_trajectory(Qs, "r-")
        plot_2d_trajectory(Qs_est, "b-")
        plt.show()

    # update state
    if accepted[i]:
        q_current = proposals[i]
    
    if accepted_est[i]:
        q_current_est = proposals_est[i]

    # store sample
    samples[i] = q_current
    samples_est[i] = q_current_est

# compute autocorrelation
acorrs = autocorr(samples[:, 0])
acorrs_est = autocorr(samples_est[:, 0])

# visualise summary
plt.figure(figsize=(8, 12))
plt.subplot(421)
plt.plot(samples[:, 0])
plt.subplot(422)
plt.plot(samples[:, 1])
plt.subplot(423)
plt.hist(samples[:, 0])
plt.subplot(424)
plt.hist(samples[:, 1])
plt.subplot(425)
plt.plot(np.cumsum(accepted) / np.arange(1, num_iterations + 1))
plt.subplot(426)
plt.plot(acorrs)
plt.subplot(427)
plt.plot(samples[:,0], samples[:,1])
plt.subplot(428)
plt.plot(samples[:,0], samples[:,1], '.')

plt.figure(figsize=(8, 12))
plt.subplot(421)
plt.plot(samples_est[:, 0])
plt.subplot(422)
plt.plot(samples_est[:, 1])
plt.subplot(423)
plt.hist(samples_est[:, 0])
plt.subplot(424)
plt.hist(samples_est[:, 1])
plt.subplot(425)
plt.plot(np.cumsum(accepted_est) / np.arange(1, num_iterations + 1))
plt.subplot(426)
plt.plot(acorrs_est)
plt.subplot(427)
plt.plot(samples_est[:,0], samples_est[:,1])
plt.subplot(428)
plt.plot(samples_est[:,0], samples_est[:,1], '.')

# discard warmup and thin
samples = samples[np.arange(num_warmup, num_iterations, step=thin)]
samples_est = samples_est[np.arange(num_warmup, num_iterations, step=thin)]

# print quantile summary
print("Empirical quantiles")
quantiles = np.arange(0.1, 1., 0.1)
q_table = PrettyTable(["%.2f" % q for q in quantiles])
quantiles_emp = emp_quantiles(samples, mu=np.zeros(D), Sigma=L.dot(L.T), quantiles=quantiles)
quantiles_emp_est = emp_quantiles(samples_est, mu=np.zeros(D), Sigma=L.dot(L.T), quantiles=quantiles)
q_table.add_row(quantiles_emp)
q_table.add_row(quantiles_emp_est)
print(q_table)

print("Quantile errors")
q_error_table = PrettyTable(["%.2f" % q for q in quantiles])
q_error_table.add_row(np.abs(quantiles_emp - quantiles))
q_error_table.add_row(np.abs(quantiles_emp_est - quantiles))
print(q_error_table)
avg_q_error = np.mean(np.abs(quantiles_emp - quantiles))
avg_q_error_est = np.mean(np.abs(quantiles_emp_est - quantiles))
print("Average quantile errors HMC: %.2f" % avg_q_error)
print("Average quantile errors KMC: %.2f" % avg_q_error_est)


plt.show()
