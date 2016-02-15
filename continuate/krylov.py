# -*- coding: utf-8 -*-
""" Krylov subspace methods

Options
--------
krylov_tol : float
    Tolerrance of Krylov iteration
krylov_maxiter : float
    Max iteration number of Krylov iteration

Their default values are set in :py:data:`.default_options`
"""

import numpy as np
from itertools import count as icount
from .logger import Logger
from . import qr

default_options = {
    "krylov_tol": 1e-9,
    "krylov_maxiter": 1e+3,
}
""" default values of options

You can get these values through :py:func:`continuate.get_default_options`
"""


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

    logger = Logger(__name__, "Arnoldi")

    def __init__(self, A, b, krylov_tol, dot=np.dot, **opt):
        self.A = A
        self.dot = dot
        self.ortho = qr.MGS(eps=krylov_tol, dot=dot)
        self.ortho(b)
        self.eps = krylov_tol
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
            coef = self.ortho(Av)
            self.residual *= coef[-1]
            self.logger.info({
                "count": c,
                "residual": self.residual,
            })
            self.coefs.append(coef)
            if self.residual < self.eps:
                self.logger.info({"matrix_norm": self.matrix_norm, })
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


def arnoldi(A, b, **opt):
    O = Arnoldi(A, b, **opt)
    return O()


def solve_Hessenberg(H, b):
    N = len(H)
    g = np.zeros((N, 1))
    g[0, 0] = b
    if N == 1:
        return g[:, 0] / H[0, 0]
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


def gmres(A, b, dot=np.dot, **opt):
    H, V = arnoldi(A, b, dot=dot, **opt)
    g = solve_Hessenberg(H, norm(b, dot=dot))
    return dot(V[:, :len(g)], g)
