

from nose.tools import timed, assert_almost_equal, assert_equal
from numpy.ma.testutils import assert_close
from numpy.testing.utils import assert_allclose

from kmc.score_matching.gaussian_rkhs import _compute_b_sym, _compute_b_low_rank_sym, \
    _compute_C_sym, _apply_left_C_sym_low_rank, score_matching_sym, \
    score_matching_sym_low_rank, _objective_sym, _objective_sym_low_rank, \
    _compute_b, _compute_C, score_matching, _apply_left_C_low_rank, \
    _compute_b_low_rank, _objective_low_rank, _objective
from kmc.score_matching.kernel.incomplete_cholesky import incomplete_cholesky_gaussian, \
    incomplete_cholesky_new_points, incomplete_cholesky
from kmc.score_matching.kernel.kernels import gaussian_kernel
import numpy as np


def test_broadcast_diag_matrix_multiply():
    K = np.random.randn(3, 3)
    x = np.array([1, 2, 3])
    assert(np.allclose(x[:, np.newaxis] * K , np.diag(x).dot(K)))

def test_score_matching_sym():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    # only run here
    a = score_matching_sym(Z, sigma, lmbda)
    assert type(a) == np.ndarray
    assert len(a.shape) == 1
    assert len(a) == len(Z)

def test_compute_b_sym_low_rank_matches_full():
    sigma = 1.
    Z = np.random.randn(100, 2)
    low_rank_dim = int(len(Z) * .9)
    K = gaussian_kernel(Z, sigma=sigma)
    R = incomplete_cholesky_gaussian(Z, sigma, eta=low_rank_dim)["R"]
    
    x = _compute_b_sym(Z, K, sigma)
    y = _compute_b_low_rank_sym(Z, R.T, sigma)
    assert_allclose(x, y, atol=5e-1)

def test_compute_b_low_rank_matches_full():
    sigma = 1.
    X = np.random.randn(100, 2)
    Y = np.random.randn(50, 2)
    
    low_rank_dim = int(len(X) * 0.9)
    kernel = lambda X, Y: gaussian_kernel(X, Y, sigma=sigma)
    K_XY = kernel(X, Y)
    temp = incomplete_cholesky(X, kernel, eta=low_rank_dim)
    I, R, nu = (temp["I"], temp["R"], temp["nu"])
    R_test = incomplete_cholesky_new_points(X, Y, kernel, I, R, nu)
    
    x = _compute_b(X, Y, K_XY, sigma)
    y = _compute_b_low_rank(X, Y, R.T, R_test.T, sigma)
    assert_allclose(x, y, atol=5e-1)

def test_compute_b_low_rank_matches_sym():
    sigma = 1.
    X = np.random.randn(10, 2)
    R = incomplete_cholesky_gaussian(X, sigma, eta=0.1)["R"]
    
    x = _compute_b_low_rank(X, X, R.T, R.T, sigma)
    y = _compute_b_low_rank_sym(X, R.T, sigma)
    assert_allclose(x, y)
    
def test_compute_b_matches_sym():
    sigma = 1.
    Z = np.random.randn(10, 2)
    
    K = gaussian_kernel(Z, sigma=sigma)
    b = _compute_b_sym(Z, K, sigma=sigma)
    b_sym = _compute_b(Z, Z, K, sigma=sigma)
    assert_allclose(b, b_sym)

def test_apply_C_left_sym_low_rank_matches_full():
    sigma = 1.
    N = 10
    Z = np.random.randn(N, 2)
    K = gaussian_kernel(Z, sigma=sigma)
    R = incomplete_cholesky_gaussian(Z, sigma, eta=0.1)["R"]
    
    v = np.random.randn(Z.shape[0])
    lmbda = 1.
    
    x = (_compute_C_sym(Z, K, sigma) + np.eye(N)).dot(v)
    y = _apply_left_C_sym_low_rank(v, Z, R.T, lmbda)
    assert_allclose(x, y, atol=1e-1)

def test_apply_C_left_low_rank_matches_full():
    sigma = 1.
    N = 100
    X = np.random.randn(N, 2)
    Y = np.random.randn(20, 2)
    low_rank_dim = int(len(X) * .9)
    kernel = lambda X, Y = None: gaussian_kernel(X, Y, sigma=sigma)
    K_XY = kernel(X, Y)
    
    temp = incomplete_cholesky(X, kernel, eta=low_rank_dim)
    I, R, nu = (temp["I"], temp["R"], temp["nu"])
    R_test = incomplete_cholesky_new_points(X, Y, kernel, I, R, nu)
    
    v = np.random.randn(X.shape[0])
    lmbda = 1.
    
    x = (_compute_C(X, Y, K_XY, sigma) + np.eye(N) * lmbda).dot(v)
    y = _apply_left_C_low_rank(v, X, Y, R.T, R_test.T, lmbda)
    assert_allclose(x, y, atol=1e-1)

def apply_C_low_rank_matches_sym():
    sigma = 1.
    N_X = 100
    X = np.random.randn(N_X, 2)
    
    kernel = lambda X, Y: gaussian_kernel(X, Y, sigma=sigma)
    temp = incomplete_cholesky(X, kernel, eta=0.1)
    I, R, nu = (temp["I"], temp["R"], temp["nu"])
    
    R_test = incomplete_cholesky_new_points(X, X, kernel, I, R, nu)
    
    v = np.random.randn(N_X.shape[0])
    lmbda = 1.
    
    x = _apply_left_C_low_rank(v, X, X, R.T, R_test.T, lmbda)
    y = _apply_left_C_sym_low_rank(v, X, R.T, lmbda)
    assert_allclose(x, y)
    
def test_compute_C_matches_sym():
    sigma = 1.
    Z = np.random.randn(10, 2)
    
    K = gaussian_kernel(Z, sigma=sigma)
    C = _compute_C_sym(Z, K, sigma=sigma)
    C_sym = _compute_C(Z, Z, K, sigma=sigma)
    assert_allclose(C, C_sym)

def test_compute_C_run_asym():
    sigma = 1.
    X = np.random.randn(100, 2)
    Y = np.random.randn(100, 2)

    K_XY = gaussian_kernel(X, Y, sigma=sigma)
    _ = _compute_C(X, Y, K_XY, sigma=sigma)

def test_score_matching_sym_low_rank_matches_full():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    a = score_matching_sym(Z, sigma, lmbda)
    
    R = incomplete_cholesky_gaussian(Z, sigma, eta=0.1)["R"]
    a_cholesky_cg = score_matching_sym_low_rank(Z, sigma, lmbda, L=R.T)
    assert_allclose(a, a_cholesky_cg, atol=3)

def test_score_matching_matches_sym():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    a_sym = score_matching_sym(Z, sigma, lmbda)
    a = score_matching(Z, Z, sigma, lmbda)
    
    assert_allclose(a, a_sym)

@timed(5)
def test_score_matching_sym_low_rank_time():
    sigma = 1.
    lmbda = 1.
    N = 20000
    Z = np.random.randn(N, 2)
    
    R = incomplete_cholesky_gaussian(Z, sigma, eta=0.1)["R"]
    score_matching_sym_low_rank(Z, sigma, lmbda, L=R.T, cg_tol=1e-1)

def test_objective_matches_sym():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    K = gaussian_kernel(Z, sigma=sigma)
    
    alpha = np.random.randn(len(Z))
    C = _compute_C_sym(Z, K, sigma)
    b = _compute_b_sym(Z, K, sigma)
    
    K = gaussian_kernel(Z, sigma=sigma)
    J_sym = _objective_sym(Z, sigma, lmbda, alpha, K, C, b)
    J = _objective(Z, Z, sigma, lmbda, alpha, K, C, b)
    
    assert_equal(J, J_sym)

def test_score_matching_objective_matches_sym():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    K = gaussian_kernel(Z, sigma=sigma)
    _, J_sym = score_matching_sym(Z, sigma, lmbda, K,
                                              compute_objective=True)
    _, J = score_matching(Z, Z, sigma, lmbda, K,
                                              compute_objective=True)
    
    assert_allclose(J, J_sym)

def test_objective_low_rank_matches_sym():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    kernel = lambda X, Y: gaussian_kernel(X, Y, sigma=sigma)
    alpha = np.random.randn(len(Z))
    
    temp = incomplete_cholesky(Z, kernel, eta=0.1)
    I, R, nu = (temp["I"], temp["R"], temp["nu"])
    
    R_test = incomplete_cholesky_new_points(Z, Z, kernel, I, R, nu)
    
    b = _compute_b_low_rank(Z, Z, R.T, R_test.T, sigma)
    
    J_sym = _objective_sym_low_rank(Z, sigma, lmbda, alpha, R.T, b)
    J = _objective_low_rank(Z, Z, sigma, lmbda, alpha, R.T, R_test.T, b)
    
    assert_allclose(J, J_sym)

def test_objective_low_rank_matches_full():
    sigma = 1.
    lmbda = 1.
    X = np.random.randn(100, 2)
    Y = np.random.randn(10, 2)
    low_rank_dim = int(len(X) * 0.9)
    
    kernel = lambda X, Y: gaussian_kernel(X, Y, sigma=sigma)
    alpha = np.random.randn(len(X))
    
    K_XY = kernel(X, Y)
    C = _compute_C(X, Y, K_XY, sigma)
    b = _compute_b(X, Y, K_XY, sigma)
    J_full = _objective(X, Y, sigma, lmbda, alpha, K_XY, C, b)
    
    temp = incomplete_cholesky(X, kernel, eta=low_rank_dim)
    I, R, nu = (temp["I"], temp["R"], temp["nu"])
    R_test = incomplete_cholesky_new_points(X, Y, kernel, I, R, nu)
    b = _compute_b_low_rank(X, Y, R.T, R_test.T, sigma)
    J = _objective_low_rank(X, Y, sigma, lmbda, alpha, R.T, R_test.T, b)
    
    assert_close(J, J_full, decimal=1)

def test_objective_sym_optimum():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    K = gaussian_kernel(Z, sigma=sigma)
    _, J_opt = score_matching_sym(Z, sigma, lmbda, K,
                                              compute_objective=True)
    
    a = np.random.randn(len(Z))
    J = _objective_sym(Z, sigma, lmbda, a, K)
    assert J >= J_opt

def test_objective_sym_same_as_from_estimation():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    K = gaussian_kernel(Z, sigma=sigma)
    a, J = score_matching_sym(Z, sigma, lmbda, K,
                                          compute_objective=True)
    
    J2 = _objective_sym(Z, sigma, lmbda, a, K)
    assert_almost_equal(J, J2)


def test_objective_sym_low_rank_same_as_from_estimation():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    L = incomplete_cholesky_gaussian(Z, sigma, eta=0.1)["R"].T
    a, J = score_matching_sym_low_rank(Z, sigma, lmbda, L,
                                                          compute_objective=True)
    
    J2 = _objective_sym_low_rank(Z, sigma, lmbda, a, L)
    assert_almost_equal(J, J2)


def test_objective_sym_low_rank_optimum():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    L = incomplete_cholesky_gaussian(Z, sigma, eta=0.1)["R"].T
    _, J_opt = score_matching_sym_low_rank(Z, sigma, lmbda, L,
                                                          compute_objective=True)
    
    a = np.random.randn(len(Z))
    J = _objective_sym_low_rank(Z, sigma, lmbda, a, L)
    assert J >= J_opt

def test_objective_sym_low_rank_matches_full():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    K = gaussian_kernel(Z, sigma=sigma)
    a_opt = score_matching_sym(Z, sigma, lmbda, K)
    J_opt = _objective_sym(Z, sigma, lmbda, a_opt, K)
    
    L = incomplete_cholesky_gaussian(Z, sigma, eta=0.01)["R"].T
    a_opt_chol = score_matching_sym_low_rank(Z, sigma, lmbda, L)
    J_opt_chol = _objective_sym_low_rank(Z, sigma, lmbda, a_opt_chol, L)
    
    assert_almost_equal(J_opt, J_opt_chol, delta=2.)

# def test_xvalidate_sigmas():
#     lmbda = 1.
#     Z = np.random.randn(100, 2)
#     
#     sigmas = np.ones(3)
#     Js = xvalidate_sigmas(Z, sigmas, lmbda)
#     
#     assert len(Js) == len(sigmas)
#     assert np.all(Js == Js[0])
