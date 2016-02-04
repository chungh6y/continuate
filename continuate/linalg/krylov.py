# -*- coding: utf-8 -*-

import numpy as np
from . import qr

from logging import getLogger, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)


class Arnoldi(object):

    def __init__(self, A, b, eps=1e-6, initialize=True, dot=np.dot):
        self.A = A
        self.dot = dot
        self.ortho = qr.MGS(eps=eps, dot=dot)
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
        return np.stack(self).T

    def calc(self):
        while True:
            Av = self.A * self.ortho[-1]
            coef = self.ortho(Av)
            logger.debug("Residual of Arnoldi iteration = {}".format(coef[-1]))
            self.coefs.append(coef)
            if coef[-1] < self.eps:
                return

    def projected_matrix(self):
        N = len(self.coefs)
        H = np.zeros((N, N))
        for i, c in enumerate(self.coefs):
            n = min(N, len(c))
            H[:n, i] = c[:n]
        return H


def arnoldi(A, b, **kwds):
    A = Arnoldi(A, b, **kwds)
    return A.projected_matrix(), A.basis()


def solve_Hessenberg(H, b):
    N = len(H)
    g = np.zeros((N, 1))
    g[0, 0] = b
    Hg = np.concatenate((H, g), axis=1)
    for i in range(N):
        Hg[i, i+1:] /= Hg[i, i]
        Hg[i, i] = 1
        if i == N-1:
            break
        Hg[i+1, i:] -= Hg[i+1, i] * Hg[i, i:]
    for i in reversed(range(1, N)):
        Hg[:i, N] -= Hg[i, N] * Hg[:i, i]
        Hg[:i, i] = 0
    return Hg[:, N]


def gmres(A, b, eps=1e-6, dot=np.dot):
    H, V = arnoldi(A, b, eps=eps, dot=dot)
    g = solve_Hessenberg(H, np.sqrt(dot(b, b)))
    return dot(V, g)
