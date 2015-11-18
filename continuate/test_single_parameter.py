# coding=utf-8

from . import single_parameter
import numpy as np
import unittest


class TestSingleParamter(unittest.TestCase):

    def test_tangent_vector_norm(self):
        """
        Norm of tangent vector is 1
        """
        for _ in range(10):
            N = 10
            A = np.random.random((N, N))
            b = np.random.random(N)
            f = lambda x, mu: np.dot(A, x) + mu * b
            x = np.random.random(N)
            mu = 2 * np.random.rand() - 1
            dx, dmu = single_parameter.tangent_vector(f, x, mu)
            self.assertTrue(np.allclose(np.dot(dx, dx) + dmu * dmu, 1))
            self.assertTrue(dmu > 0)
