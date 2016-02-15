# -*- coding: utf-8 -*-

from . import krylov

import numpy as np
import scipy.sparse.linalg as linalg
from unittest import TestCase


class TestKrylov(TestCase):

    def setUp(self):
        self.N = 5
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

    def _unit_vector(self, i):
        return np.identity(self.N)[:, i]

    def test_iterate_random(self):
        """ Check Arnold.iteration for random matrix

        Iteration must be continue until Arnoldi.H becomes square
        """
        A = self._random_linop()
        b = self._random_vector()
        O = krylov.Arnoldi(A, b, **self.opt)
        H = O.projected_matrix()
        self.assertEqual(H.shape, (self.N, self.N))

    def test_iterate_identity(self):
        """ Check Arnold.iteration for identity matrix

        Iteration does not creates Krylov subspace
        """
        A = linalg.aslinearoperator(np.identity(self.N))
        b = self._unit_vector(0)
        O = krylov.Arnoldi(A, b, **self.opt)
        H = O.projected_matrix()
        self.assertEqual(H.shape, (1, 1))
        np.testing.assert_equal(H[0, 0], 1)

    def test_basis(self):
        """ Check orthogonality of the basis """
        A = self._random_linop()
        b = self._random_vector()
        O = krylov.Arnoldi(A, b, **self.opt)
        V = O.basis()
        self.assertEqual(V.shape, (self.N, self.N))
        I = np.identity(self.N)
        np.testing.assert_almost_equal(np.dot(V.T, V), I)
        np.testing.assert_almost_equal(np.dot(V, V.T), I)
        for i, v in enumerate(O):
            np.testing.assert_almost_equal(V[:, i], v)
        for i in range(self.N):
            vi = V[:, i]
            np.testing.assert_almost_equal(np.dot(vi, vi), 1.0)
            for j in range(i + 1, self.N):
                vj = V[:, j]
                np.testing.assert_array_almost_equal(np.dot(vi, vj), 0.0)

    def test_AV_VH(self):
        """ Confirm AV = VH """
        A = self._random_linop()
        b = self._random_vector()
        H, V = krylov.arnoldi(A, b, **self.opt)
        np.testing.assert_almost_equal(A * V, np.dot(V, H))

    def _hessenberg_matrix(self):
        rand = np.random.rand(self.N, self.N)
        for i in range(self.N):
            rand[i+1:, :i] = 0
        return rand

    def test_hessenberg(self):
        """ Case where A is Hessenberg matrix

        - V becomes identity matrix when A is Hessenberg.
        - Thus, H equals to A.
        """
        rand = self._hessenberg_matrix()
        A = linalg.aslinearoperator(rand)
        b = self._unit_vector(0)
        H, V = krylov.arnoldi(A, b, **self.opt)
        np.testing.assert_almost_equal(H, rand)
        np.testing.assert_almost_equal(V, np.identity(self.N))

    def test_solve_Hessenberg(self):
        h = self._hessenberg_matrix()
        A = linalg.aslinearoperator(h)
        g = krylov.solve_Hessenberg(h, 1)
        c = self._unit_vector(0)
        np.testing.assert_almost_equal(np.dot(h, g), c)

    def test_gmres(self):
        A = self._random_linop()
        x = self._random_vector()
        b = A * x
        y = krylov.gmres(A, b, **self.opt)
        np.testing.assert_almost_equal(y, x)
