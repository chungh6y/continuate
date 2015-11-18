# -*- coding: utf-8 -*-

from . import krylov

import numpy as np
import scipy.sparse.linalg as linalg
from unittest import TestCase


class TestKrylov(TestCase):

    def test_iterate_random(self):
        """ Check Arnold.iteration for random matrix

        Iteration must be continue until Arnoldi.H becomes square
        """
        N = 5
        rand = np.random.rand(N, N)
        A = linalg.LinearOperator((N, N), matvec=lambda x: np.dot(rand, x), dtype=np.float64)
        b = np.array([2], dtype=np.float64)
        b.resize(N)
        O = krylov.Arnoldi(A, b)
        for i in range(N - 1):
            self.assertIsNotNone(O.iterate())
        self.assertIsNone(O.iterate())
        H = O.get_projected_matrix()
        self.assertEqual(H.shape, (N, N))

    def test_iterate_identity(self):
        """ Check Arnold.iteration for identity matrix

        Iteration does not creates Krylov subspace
        """
        N = 5
        I = np.identity(N)
        A = linalg.LinearOperator((N, N), matvec=lambda x: np.dot(I, x), dtype=np.float64)
        b = np.array([2], dtype=np.float64)
        b.resize(N)
        O = krylov.Arnoldi(A, b)
        self.assertIsNone(O.iterate())
        H = O.get_projected_matrix()
        self.assertEqual(H.shape, (N, 1))
        Ha = np.zeros_like(H)
        Ha[0, 0] = 1
        np.testing.assert_equal(H, Ha)

    def test_over_iterate(self):
        N = 5
        rand = np.random.rand(N, N)
        A = linalg.LinearOperator((N, N), matvec=lambda x: np.dot(rand, x), dtype=np.float64)
        b = np.array([2], dtype=np.float64)
        b.resize(N)
        O = krylov.Arnoldi(A, b)
        while O.iterate() is not None:
            pass
        self.assertIsNone(O.iterate())
        H = O.get_projected_matrix()
        self.assertEqual(H.shape, (N, N))

    def test_basis(self):
        """ Check orthogonality of the basis """
        N = 5
        rand = np.random.rand(N, N)
        A = linalg.LinearOperator((N, N), matvec=lambda x: np.dot(rand, x), dtype=np.float64)
        b = np.array([2], dtype=np.float64)
        b.resize(N)
        _, V = krylov.arnoldi(A, b)
        self.assertEqual(V.shape, (N, N))
        v0 = np.zeros(N)
        v0[0] = 1.0
        np.testing.assert_equal(V[:, 0], v0)
        for i in range(N):
            vi = V[:, i]
            np.testing.assert_almost_equal(np.dot(vi, vi), 1.0)
            for j in range(i + 1, N):
                vj = V[:, j]
                np.testing.assert_array_almost_equal(np.dot(vi, vj), 0.0)

    def test_h(self):
        """ Confirm AV = VH """
        N = 5
        rand = np.random.rand(N, N)
        A = linalg.LinearOperator((N, N), matvec=lambda x: np.dot(rand, x), dtype=np.float64)
        b = np.zeros(N)
        b[0] = 2.0
        H, V = krylov.arnoldi(A, b)
        np.testing.assert_almost_equal(A * V, np.dot(V, H))
