from matplotlib.widgets import Slider, Button, RadioButtons

from kmc.densities.gaussian import log_gaussian_pdf, sample_gaussian
from kmc.densities.student import log_student_pdf
from kmc.score_matching.random_feats.gaussian_rkhs import xvalidate, compute_b,\
    compute_C, score_matching_sym, feature_map_single, feature_map_grad_single,\
    objective
import matplotlib.pyplot as plt
import numpy as np
from scripts.tools.plotting import evaluate_density_grid,\
    evaluate_density_grad_grid, plot_array


plot_pdf = True

def plot_lmbda_surface(val):
    print("lambda")
    log2_sigma = s_sigma.val
     
    log2_lambdas = np.linspace(s_lmbda.valmin, s_lmbda.valmax, 50)
    Js = np.array([np.mean(xvalidate(Z, 2**log2_lmbda, omega, u, n_folds=5, num_repetitions=3)) for log2_lmbda in log2_lambdas])
     
    log2_lambda_min = log2_lambdas[Js.argmin()]
    log_Js = np.log(Js - (Js.min() if Js.min() < 0 else 0) + 1)
 
    # update slider
    s_lmbda.set_val(log2_lambda_min)
    update_plot()
     
    plt.figure()
    plt.plot(log2_lambdas, log_Js)
    plt.plot([log2_lambda_min, log2_lambda_min], [log_Js.min(), log_Js.max()], 'r')
    plt.title(r"$\lambda$ surface for $\log_2 \sigma=%.2f$, best value of $J(\alpha)=%.2f$ at $\log_2 \lambda=%.2f$" % 
              (log2_sigma, Js.min(), log2_lambda_min))
     
    plt.show()

def optimise_sigma_surface(val):
    global gamma
    print("sigma")
    log2_lmbda = s_lmbda.val
    lmbda = 2**log2_lmbda
     
    log2_sigmas = np.linspace(s_sigma.valmin, s_sigma.valmax, 50)
    Js = np.zeros(len(log2_sigmas))
    for i,log2_sigma in enumerate(log2_sigmas):
        s_sigma.val=log2_sigma
        update_basis(val)
        Js[i] = np.mean(xvalidate(Z, lmbda, omega, u, n_folds=5, num_repetitions=3))
     
    log2_sigma_min = log2_sigmas[Js.argmin()]
    log_Js = np.log(Js - (Js.min() if Js.min() < 0 else 0) + 1)
     
    # update slider
    s_sigma.set_val(log2_sigma_min)
    update_plot()
 
    plt.figure()
    plt.plot(log2_sigmas, log_Js)
    plt.plot([log2_sigma_min, log2_sigma_min], [log_Js.min(), log_Js.max()], 'r')
    plt.title(r"$\sigma$ surface for $\log_2 \lambda=%.2f$, best value of $J(\alpha)=%.2f$ at $\log_2 \sigma=%.2f$" % 
              (log2_lmbda, Js.min(), log2_sigma_min))
     
     
    plt.show()

def update_plot(val=None):
    global omega, u
    print("Updating plot")
    
    lmbda = 2 ** s_lmbda.val
    sigma = 2 ** s_sigma.val
    
    b = compute_b(Z, omega, u)
    C = compute_C(Z, omega, u)
    theta = score_matching_sym(Z, lmbda, omega, u, b, C)
    J = objective(Z, theta, lmbda, omega, u, b, C)
    J_xval = np.mean(xvalidate(Z, lmbda, omega, u, n_folds=5, num_repetitions=3))
    
    logq_est = lambda x: np.dot(theta, feature_map_single(x, omega, u))
    dlogq_est = lambda x: np.dot(theta, feature_map_grad_single(x, omega, u))

    description = "N=%d, sigma: %.2f, lambda: %.2f, m=%.d, J=%.2f, J_xval=%.2f" % \
        (N, sigma, lmbda, m, J, J_xval)
        
    if plot_pdf:
        D = evaluate_density_grid(Xs, Ys, logq_est)
        description = "log-pdf: " + description
    else:
        D = evaluate_density_grad_grid(Xs, Ys, dlogq_est)
        description = "norm-grad-log-pdf: " + description
    
    ax.clear()
    ax.plot(Z[:, 0], Z[:, 1], 'bx')
    plot_array(Xs, Ys, D, ax, plot_contour=True)
    
        
    ax.set_title(description)

    fig.canvas.draw_idle()

def plot_true():
    G = evaluate_density_grid(Xs, Ys, logq)
    G_grad = evaluate_density_grad_grid(Xs, Ys, dlogq)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plot_array(Xs, Ys, G, plot_contour=True)
    plt.plot(Z[:, 0], Z[:, 1], 'bx')
    plt.title("True log-pdf")
    
    plt.subplot(122)
    plot_array(Xs, Ys, G_grad, plot_contour=True)
    plt.plot(Z[:, 0], Z[:, 1], 'bx')
    plt.title("True gradient norm log-pdf")

def run_cma(val):
    pass
#     cma_opts = {'tolfun':1e-2, 'maxiter':20, 'verb_disp':1,
#             'bounds': [-5, 10]}
#     
#     sigma0 = 2**s_sigma.val
#     lmbda0 = 2**s_lmbda.val
#     
#     es = select_sigma_lambda_cma(Z, sigma0=sigma0, lmbda0=lmbda0,
#                                  cma_opts=cma_opts, disp=False)
#     log2_sigma = es.best.get()[0][0]
#     log2_lmbda = es.best.get()[0][1]
#     
#     s_lmbda.set_val(log2_lmbda)
#     s_sigma.set_val(log2_sigma)
#     
#     update_plot()

def radio_callback(val):
    global plot_pdf
    
    if val == "log-pdf":
        plot_pdf = True
    else:
        plot_pdf = False
    
    update_plot()
    
    
def update_basis(val):
    global omega, u, m, gamma
    sigma = 2**s_sigma.val
    gamma = 0.5/(sigma**2)
    m = int(s_m.val)
    print("Updating basis for m=%d, gamma=%.2f" % (m, gamma))
    omega = gamma * np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)

def update_basis_plot(val):
    update_basis(val)
    update_plot(val)

if __name__ == "__main__":
    D = 2
    N = 200
    m = 100
    gamma = 1.
    omega = gamma * np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    # true target log density
    Sigma = np.diag(np.linspace(0.01, 1, D))
    Sigma[:2, :2] = np.array([[1, .95], [.95, 1]])
    Sigma = np.eye(D)
    L = np.linalg.cholesky(Sigma)
    dlogq = lambda x: log_gaussian_pdf(x, Sigma=L, is_cholesky=True, compute_grad=True)
    logq = lambda x: log_gaussian_pdf(x, Sigma=L, is_cholesky=True, compute_grad=False)
    
    dlogq = lambda x: log_student_pdf(x, nu=1., compute_grad=True)
    logq = lambda x: log_student_pdf(x, nu=1., compute_grad=False)
    

    # sample density
    mu = np.zeros(D)
    np.random.seed(0)
    Z = sample_gaussian(N, mu, Sigma=L, is_cholesky=True)
#     print np.sum(Z) * np.std(Z) * np.sum(Z**2) * np.std(Z**2)

    Xs = np.linspace(-3, 3)
    Ys = np.linspace(-3, 3)
    
    plot_true()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.3)
    ax_color = 'lightgoldenrodyellow'
    ax_sigma = plt.axes([0.25, 0.2, 0.65, 0.03], axisbg=ax_color)
    ax_lmbda = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=ax_color)
    ax_m = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=ax_color)
    s_sigma = Slider(ax_sigma, 'log2 sigma', -3, 3, valinit=0)
    s_lmbda = Slider(ax_lmbda, 'log2 lambda', -3, 3., valinit=0)
    s_m = Slider(ax_m, 'm', 1, 2 * N, valinit=100)
    s_sigma.on_changed(update_basis_plot)
    s_lmbda.on_changed(update_basis_plot)
    s_m.on_changed(update_basis_plot)
    
    ax_sigma_btn = plt.axes([0.1, 0.025, 0.1, 0.04])
    btn_sigma_btn = Button(ax_sigma_btn, 'sigma', color=ax_color, hovercolor='0.975')
    btn_sigma_btn.on_clicked(optimise_sigma_surface)
    
    ax_lmbda_btn = plt.axes([0.3, 0.025, 0.1, 0.04])
    btn_lambda_btn = Button(ax_lmbda_btn, 'lambda', color=ax_color, hovercolor='0.975')
    btn_lambda_btn.on_clicked(plot_lmbda_surface)
    
    ax_cma_btn = plt.axes([0.5, 0.025, 0.1, 0.04])
    btn_cma_btn = Button(ax_cma_btn, 'cma', color=ax_color, hovercolor='0.975')
    btn_cma_btn.on_clicked(run_cma)
    
    ax_basis_btn = plt.axes([0.7, 0.025, 0.1, 0.04])
    btn_basis_opt = Button(ax_basis_btn, 'Basis', color=ax_color, hovercolor='0.975')
    btn_basis_opt.on_clicked(update_basis_plot)
    
    ax_radio = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=ax_color)
    radio = RadioButtons(ax_radio, ('log-pdf', 'norm-log-gradient'), active=0)
    radio.on_clicked(radio_callback)
    
    update_plot(0)

    
    plt.show()
