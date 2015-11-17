# -*- coding: utf-8 -*-

from . import linalg
import numpy as np
import unittest


class TestLinalgJacobi(unittest.TestCase):

    def test_Jacobi_linear(self):
        """
        Jacobi matrix of a linear function equals to the original.
        """
        shape = (10, 10)
        A = np.random.random(shape)
        f = lambda x: np.dot(A, x)
        x0 = np.zeros(shape[0])
        J = linalg.Jacobian(f, x0)
        for _ in range(10):
            x = np.random.random(shape[0])
            np.allclose(f(x), J(x))

    def test_Jacobi_polynominal(self):
        """
        Test simple nonlinear case
        """
        f = lambda x: np.array([x[1]**2, x[0]**2])
        x0 = np.array([1, 2])
        J1 = linalg.Jacobian(f, x0)
        A = np.array([[0, 2*2], [2*1, 0]])
        J2 = lambda x: np.dot(A, x)
        for _ in range(10):
            x = np.random.random(2)
            np.allclose(J1(x), J2(x))
