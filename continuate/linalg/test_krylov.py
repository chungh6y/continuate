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
        H = O.projected_matrix()
        self.assertEqual(H.shape, (N, N))

    def test_iterate_identity(self):
        """ Check Arnold.iteration for identity matrix

        Iteration does not creates Krylov subspace
        """
        N = 5
        A = linalg.LinearOperator((N, N), matvec=lambda x: x, dtype=np.float64)
        b = np.array([2], dtype=np.float64)
        b.resize(N)
        O = krylov.Arnoldi(A, b)
        H = O.projected_matrix()
        self.assertEqual(H.shape, (1, 1))
        Ha = np.zeros_like(H)
        Ha[0, 0] = 1
        np.testing.assert_equal(H, Ha)

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

    def test_hessenberg(self):
        N = 5
        rand = np.random.rand(N, N)
        for i in range(N):
            rand[i+1:, :i] = 0
        A = linalg.LinearOperator((N, N), matvec=lambda x: np.dot(rand, x), dtype=np.float64)
        b = np.zeros(N)
        b[0] = 1
        H, V = krylov.arnoldi(A, b)
        np.testing.assert_almost_equal(H, rand)
        np.testing.assert_almost_equal(V, np.identity(N))

    def test_h(self):
        """ Confirm AV = VH """
        N = 5
        rand = np.random.rand(N, N)
        A = linalg.LinearOperator((N, N), matvec=lambda x: np.dot(rand, x), dtype=np.float64)
        b = np.random.rand(N)
        O = krylov.Arnoldi(A, b)
        H = O.projected_matrix()
        V = O.basis()
        np.testing.assert_almost_equal(V[:, 0], O[0])
        np.testing.assert_almost_equal(np.dot(V.T, V), np.identity(N))
        np.testing.assert_almost_equal(np.dot(V, V.T), np.identity(N))
        np.testing.assert_almost_equal(A * V, np.dot(V, H))

    def test_solve_Hessenberg(self):
        N = 5
        rand = np.random.rand(N, N)
        A = linalg.LinearOperator((N, N), matvec=lambda x: np.dot(rand, x), dtype=np.float64)
        b = np.random.rand(N)
        O = krylov.Arnoldi(A, b)
        H = O.projected_matrix()
        g = krylov.solve_Hessenberg(H, 1)
        c = np.zeros(N)
        c[0] = 1
        np.testing.assert_almost_equal(np.dot(H, g), c)

    def test_gmres(self):
        N = 5
        rand = np.random.rand(N, N)
        A = linalg.LinearOperator((N, N), matvec=lambda x: np.dot(rand, x), dtype=np.float64)
        x = np.random.rand(N)
        b = A * x
        y = krylov.gmres(A, b)
        np.testing.assert_almost_equal(y, x)
