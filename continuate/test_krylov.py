# -*- coding: utf-8 -*-

from . import krylov

import numpy as np
import scipy.sparse.linalg as linalg
from unittest import TestCase


class TestKrylov(TestCase):

    def setUp(self):
        self.N = 100
        self.opt = krylov.default_options

    def _random_linop(self):
        rand = np.random.rand(self.N, self.N)
        return linalg.LinearOperator(
            (self.N, self.N),
            matvec=lambda x: np.dot(rand, x),
            dtype=np.float64
        )

    def _random_vector(self):
        return np.random.random(self.N)

    def test_basis(self):
        """ Check orthogonality of the basis """
        A = self._random_linop()
        b = self._random_vector()
        V = krylov.gmres_factorize(A, b)[0]
        self.assertEqual(V.shape, (self.N, self.N))
        I = np.identity(self.N)
        np.testing.assert_almost_equal(np.dot(V.T, V), I)
        np.testing.assert_almost_equal(np.dot(V, V.T), I)
        for i in range(self.N):
            vi = V[:, i]
            np.testing.assert_almost_equal(np.dot(vi, vi), 1.0)
            for j in range(i + 1, self.N):
                vj = V[:, j]
                np.testing.assert_array_almost_equal(np.dot(vi, vj), 0.0)

    def test_gmres(self):
        A = self._random_linop()
        x = self._random_vector()
        b = A * x
        y = krylov.gmres(A, b, **self.opt)
        np.testing.assert_almost_equal(y, x)
