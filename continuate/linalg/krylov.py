# -*- coding: utf-8 -*-

import numpy as np
from . import qr

from logging import getLogger, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)


class Arnoldi2(object):

    def __init__(self, A, b, eps=1e-6, initialize=True):
        self.A = A
        self.ortho = qr.MGS()
        self.ortho(b)
        self.eps = eps
        self.coefs = []
        if initialize:
            self.calc()

    def __iter__(self):
        return self.ortho.__iter__()

    def __getitem__(self, i):
        return self.ortho[i]

    def basis(self):
        return self.__iter__()

    def calc(self):
        while True:
            Av = self.A * self.ortho[-1]
            coef, u_norm = self.ortho(Av)
            logger.debug("Residual of Arnoldi iteration = {}".format(u_norm))
            self.coefs.append(coef)
            if u_norm < self.eps:
                return

    def projected_matrix(self):
        N = len(self.coefs)
        H = np.zeros((N, N))
        for i, c in enumerate(self.coefs):
            H[:len(c), i] = c
        return H


class Arnoldi(object):

    def __init__(self, A, b):
        self.A = A
        b_norm = np.linalg.norm(b)
        self.basis = [b / b_norm]
        self.H = []

    def iterate(self, e=1e-10):
        """ iterate Arnoldi process

        Parameters
        ----------
        e : float, optional (default=1e-10)
            Residual threshold

        Returns
        --------
        (residual, unit vector)

        """
        v = self.basis[-1]
        if len(self.H) >= len(v):
            return None
        u = self.A * v
        weight = []
        for b in self.basis:
            w = np.dot(b, u)
            weight.append(w)
            u -= w * b
        u_norm = np.linalg.norm(u)
        weight.append(u_norm)
        self.H.append(np.array(weight))
        if u_norm > e:
            b = u / u_norm
            self.basis.append(b)
            return (u_norm, b)
        return None

    def get_basis(self):
        N = len(self.basis[0])
        resized = [np.resize(b, (N, 1)) for b in self.basis]
        return np.concatenate(resized, axis=1)

    def get_projected_matrix(self):
        N = len(self.basis[0])
        resized = []
        for i, h in enumerate(self.H):
            tmp = np.resize(h, (N, 1))
            for j in range(i + 2, N):
                tmp[j, 0] = 0
            resized.append(tmp)
        return np.concatenate(resized, axis=1)


def arnoldi(A, b, e=1e-10):
    """ get Arnoldi projected matrix and its basis

    Parameters
    ----------
    A : scipy.sparse.linalg.LinearOperator
        linear operator
    b : array like
        basis of Krylov subspace
    e : float, optional (default=1e-10)
        Residual threshold

    Returns
    -------
    (H, V)
        H is projected Hessemberg matrix,
        and V is basis (V[:,n] is each basis vector).

    """
    O = Arnoldi(A, b)
    while O.iterate(e) is not None:
        pass
    return (O.get_projected_matrix(), O.get_basis())
