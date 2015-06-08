from numpy.testing.utils import assert_allclose

import numpy as np
from scripts.tools.SPSA import SPSA


def test_SPSA():
    loss = lambda x: 0.5 * x.dot(x)
    grad_true = lambda x: x
    
    x_test = np.array([1, 2])
    grad_est = SPSA(loss, x_test, stepsize=.5, num_repeats=1000)
    assert_allclose(grad_true(x_test), grad_est, atol=.1)