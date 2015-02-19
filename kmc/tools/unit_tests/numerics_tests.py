from numpy.testing.utils import assert_allclose
import unittest

from kmc.tools.numerics import log_sum_exp, log_mean_exp
import numpy as np

class Test(unittest.TestCase):
    def test_log_sum_exp(self):
        X=np.abs(np.random.randn(100))
        direct=np.log(np.sum(np.exp(X)))
        indirect=log_sum_exp(X)
        assert_allclose(direct, indirect)
        
    def test_log_mean_exp(self):
        X=np.abs(np.random.randn(100))
        direct=np.log(np.mean(np.exp(X)))
        indirect=log_mean_exp(X)
        assert_allclose(direct, indirect)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()