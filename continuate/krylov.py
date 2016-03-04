# -*- coding: utf-8 -*-
""" Krylov subspace methods

These methods are based on the Arnoldi process

.. math:: AV_n = V_{n+1} H_{n+1},

where :math:`V_n` denotes the basis of Krylov subspace,
and :math:`H_n` denotes the projected matrix.

Options
--------
krylov_tol : float
    Tolerrance of Krylov iteration
krylov_maxiter : float
    Max iteration number of Krylov iteration

Their default values are set in :py:data:`.default_options`
"""

import numpy as np
from numpy import dot
from numpy.linalg import norm
from itertools import count as icount
from .misc import Logger
from . import qr, exceptions

default_options = {
    "krylov_tol": 1e-9,
    "krylov_maxiter": 100,
}
""" default values of options

You can get these values through :py:func:`continuate.get_default_options`
"""


def arnoldi_common(A, r, krylov_tol=default_options["krylov_tol"],
                   krylov_maxiter=default_options["krylov_maxiter"], **cfg):
    """ Support generator for Arnoldi process

    Parameters
    -----------
    A : scipy.sparse.linalg.LinearOperator
        :code:`*` operator is needed.
    r : np.array
        The base of Krylov subspace :math:`K = \\left<r, Ar, A^2r, ...\\right>`

    Yields
    -------
    V : np.array (2d)
        With shape :math:`(N, n)`
    h : np.array (1d)
        The last column of :math:`H_{n+1}`

    """
    mgs = qr.MGS(eps=krylov_tol)
    mgs(r)
    for n in range(krylov_maxiter):
        v = mgs[-1]
        Av = A*v
        h = mgs(Av)
        yield mgs.V.T, h
    raise exceptions.MaxIteration("arnoldi_common")


def gmres_gen(A, b, x0=None, krylov_tol=default_options["krylov_tol"],
              krylov_maxiter=default_options["krylov_maxiter"]):
    """ Solve linear equations :math:`Ax=b` by GMRES

    Parameters
    -----------
    A : scipy.sparse.linalg.LinearOperator
        :code:`*` operator is needed.
    b : np.array
        inhomogeneous term
    x0 : np.array
        Initial guess of linear problem

    Examples
    ----------
    >>> from numpy.random import random
    >>> from scipy.sparse.linalg import aslinearoperator
    >>> A = aslinearoperator(random((5, 5)))
    >>> x = random(5)
    >>> b = A*x
    >>> ans = gmres_gen(A, b)
    >>> np.allclose(ans, x)
    True

    """
    logger = Logger(__name__, "GMRES")
    if x0 is None:
        r = b
    else:
        r = b - A*x0
    G = arnoldi_common(A, r, krylov_tol=krylov_tol,
                       krylov_maxiter=krylov_maxiter)
    Q = []
    hs = []
    g = np.array([norm(r)])
    for V, h in G:
        g = np.append(g, 0)
        for n, q in enumerate(Q):
            h[n:n+2] = np.dot(q, h[n:n+2])
        q = np.array([[h[-2], h[-1]], [-h[-1], h[-2]]]) / norm(h[-2:])
        Q.append(q)
        h[-2:] = np.dot(q, h[-2:])
        g[-2:] = np.dot(q, g[-2:])
        hs.append(h[:-1])
        logger.info({"count": len(h)-2, "residual": np.abs(g[-1])})
        if np.abs(g[-1]) < krylov_tol:
            break
    H = np.zeros((len(hs), len(hs)))
    for n, h in enumerate(hs):
        H[:n+1, n] = h
    y = np.linalg.solve(H, g[:-1])
    return np.dot(V, y)


class Arnoldi(object):
    """ Construct Krylov subspace (Arnoldi process)

    .. math::
        AV_n = V_n H_n + re^T

    Attributes
    -----------
    residual : float
        The residual of Arnoldi Process
    matrix_norm : float
        Approximated (projected) matrix norm of `A`

    """

    logger = Logger(__name__, "Arnoldi")

    def __init__(self, A, b, krylov_tol, **opt):
        self.A = A
        self.ortho = qr.MGS(eps=krylov_tol)
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
            self.matrix_norm = max(self.matrix_norm, norm(Av))
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


def gmres(A, b, **opt):
    H, V = arnoldi(A, b, **opt)
    g = solve_Hessenberg(H, norm(b))
    return dot(V[:, :len(g)], g)
