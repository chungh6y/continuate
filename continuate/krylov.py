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
from numpy.linalg import norm
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


def gmres(A, b, x0=None, krylov_tol=default_options["krylov_tol"],
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
    >>> ans = gmres(A, b)
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
