# -*- coding: utf-8 -*-

import numpy as np
from itertools import count as icount
from . import qr

from logging import getLogger, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)


def norm(v, dot=np.dot):
    return np.sqrt(dot(v, v))


class Arnoldi(object):
    """ Construct Krylov subspace (Arnoldi process)

    Attributes
    -----------
    residual : float
        The residual of Arnoldi Process
    matrix_norm : float
        Approximated (projected) matrix norm of `A`

    """

    def __init__(self, A, b, eps=1e-6, dot=np.dot):
        self.A = A
        self.dot = dot
        self.ortho = qr.MGS(eps=eps, dot=dot)
        self.ortho(b)
        self.eps = eps
        self.coefs = []
        self._calc()

    def __iter__(self):
        return self.ortho.__iter__()

    def __getitem__(self, i):
        return self.ortho[i]

    def _calc(self):
        """ Main process of Arnoldi process """
        self.residual = 1.0
        self.matrix_norm = 0.0
        for c in icount():
            v = self.ortho[-1]
            Av = self.A * v
            norm_Av = norm(Av, dot=self.dot)
            self.matrix_norm = max(self.matrix_norm, norm_Av)
            logger.debug("|Av|={}".format(norm_Av))
            logger.debug("(v, Av)/|Av|={}".format(self.dot(v, Av) / norm_Av))
            coef = self.ortho(Av)
            self.residual *= coef[-1] / norm(coef, dot=self.dot)
            logger.info("Arnoldi: Count={}, Residual={}"
                        .format(c, self.residual))
            self.coefs.append(coef)
            if self.residual < self.eps:
                logger.info("Matrix Norm={}".format(self.matrix_norm))
                return

    def basis(self):
        return np.stack(self).T

    def projected_matrix(self):
        N = len(self.coefs)
        H = np.zeros((N, N))
        for i, c in enumerate(self.coefs):
            n = min(N, len(c))
            H[:n, i] = c[:n]
        return H

    def __call__(self):
        return self.projected_matrix(), self.basis()


def arnoldi(A, b, **kwds):
    O = Arnoldi(A, b, **kwds)
    return O()


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
    g = solve_Hessenberg(H, norm(b, dot=dot))
    return dot(V[:, :len(g)], g)
