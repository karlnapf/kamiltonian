from numpy.ma.testutils import assert_close
from numpy.testing.utils import assert_allclose

from kmc.score_matching.random_feats.gaussian_rkhs import feature_map_single, \
    feature_map, feature_map_derivative_d, feature_map_derivative2_d, \
    feature_map_derivatives_loop, feature_map_derivatives2_loop, \
    feature_map_derivatives2, feature_map_derivatives, compute_b, compute_C
import numpy as np


def test_feature_map():
    x = 3.
    u = 2.
    omega = 2.
    phi = feature_map_single(x, omega, u)
    phi_manual = np.cos(omega * x + u) * np.sqrt(2.)
    assert_close(phi, phi_manual)

def test_feature_map_single_equals_feature_map():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    phis = feature_map(X, omega, u)
    
    for i, x in enumerate(X):
        phi = feature_map_single(x, omega, u)
        assert_allclose(phis[i], phi)

def test_feature_map_derivative_d():
    X = np.array([[1.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    phi_derivative = feature_map_derivative_d(X, omega, u, d)
    phi_derivative_manual = -np.sin(X * omega + u) * omega[:, d] * np.sqrt(2.)
    assert_close(phi_derivative, phi_derivative_manual)

def test_feature_map_derivative2_d():
    X = np.array([[1.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    phi_derivative2 = feature_map_derivative2_d(X, omega, u, d)
    phi_derivative2_manual = feature_map(X, omega, u) * (omega[:, d] ** 2)
    assert_close(phi_derivative2, phi_derivative2_manual)

def test_feature_map_derivatives_loop_equals_map_derivative_d():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = feature_map_derivatives_loop(X, omega, u)
    
    for d in range(D):
        derivative = feature_map_derivative_d(X, omega, u, d)
        assert_allclose(derivatives[d], derivative)

def test_feature_map_derivatives_equals_feature_map_derivatives_loop():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = feature_map_derivatives(X, omega, u)
    derivatives_loop = feature_map_derivatives_loop(X, omega, u)
    
    assert_allclose(derivatives_loop, derivatives)

def test_feature_map_derivatives2_loop_equals_map_derivative2_d():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = feature_map_derivatives2_loop(X, omega, u)
    
    for d in range(D):
        derivative = feature_map_derivative2_d(X, omega, u, d)
        assert_allclose(derivatives[d], derivative)

def test_feature_map_derivatives2_equals_feature_map_derivatives2_loop():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = feature_map_derivatives2(X, omega, u)
    derivatives_loop = feature_map_derivatives2_loop(X, omega, u)
    
    assert_allclose(derivatives_loop, derivatives)

def test_compute_b():
    X = np.array([[1.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    b_manual = feature_map_derivative_d(X, omega, u, d).flatten()
    b = compute_b(X, omega, u)
    assert_allclose(b_manual, b)

def test_compute_C():
    X = np.array([[1.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    phi = feature_map_derivative2_d(X, omega, u, d).flatten()
    C_manual = np.outer(phi, phi)
    C = compute_C(X, omega, u)
    assert_allclose(C_manual, C)
