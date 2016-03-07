# -*- coding: utf-8 -*-
""" Krylov subspace methods

These methods are based on the Arnoldi process

.. math:: AV_n = V_{n+1} H_{n+1},

where :math:`V_n` denotes the basis of Krylov subspace,
and :math:`H_n` denotes the projected matrix with Hessenberg form.

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
from . import qr

default_options = {
    "krylov_tol": 1e-9,
    "krylov_maxiter": 100,
}
""" default values of options

You can get these values through :py:func:`continuate.get_default_options`
"""


def arnoldi_common(A, r, krylov_tol=default_options["krylov_tol"], **cfg):
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
        With shape :math:`(N, n)` (`n` starts with `2`)
    h : np.array (1d)
        The last column of :math:`H_n` with shape `(n, )`
    """
    K = [r]
    G = qr.mgs(K, eps=krylov_tol)
    for n, (V, h) in enumerate(G):
        v = V[:, -1]
        K.append(A*v)
        if n == 0:
            continue
        if len(h) == n:
            yield V, np.append(h, 0)
            return
        else:
            yield V, h


def gmres_factorize(A, b, x0=None, krylov_tol=default_options["krylov_tol"],
                    krylov_maxiter=default_options["krylov_maxiter"], **cfg):
    """
    Execute a factorization :math:`AV = VQR`
    to solve minimization problem :math:`|Ax-b| = |VQ(Ry-g)| = |Ry-g|`.

    Parameters
    -----------
    A : scipy.sparse.linalg.LinearOperator
        :code:`*` operator is needed.
    b : np.array
        inhomogeneous term
    x0 : np.array
        Initial guess of linear problem

    Returns
    --------
    V : np.array (2d)
        The basis of Krylov subspace with shape :math:`(N, n)`
    R : np.array (2d)
        Projected, and QR-decomposed matrix :math:`AV = VQR`
    g : np.array (1d)
        Q-rotated vector of :math:`b`
    Q : [np.array(2x2)]
        Givens rotation matrix

    Examples
    ----------
    >>> from numpy.random import random
    >>> from scipy.sparse.linalg import aslinearoperator

    >>> A = aslinearoperator(random((5, 5)))
    >>> b = random(5)
    >>> V, R, g, Q = gmres_factorize(A, b)
    >>> V.shape, R.shape, g.shape
    ((5, 5), (5, 5), (5,))

    >>> A = aslinearoperator(np.identity(5))
    >>> V, R, g, Q = gmres_factorize(A, b)
    >>> V.shape, R.shape, g.shape
    ((5, 1), (1, 1), (1,))

    """
    logger = Logger(__name__, "GMRES")
    if x0 is None:
        r = b
    else:
        r = b - A*x0
    G = arnoldi_common(A, r, krylov_tol=krylov_tol)
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
    return V, H, g[:-1], Q


def gmres(A, b, x0=None, krylov_tol=default_options["krylov_tol"],
          krylov_maxiter=default_options["krylov_maxiter"], **cfg):
    """ Solve linear equations :math:`Ax=b` by GMRES

    Parameters
    -----------
    A : scipy.sparse.linalg.LinearOperator
        :code:`*` operator is needed.
    b : np.array
        inhomogeneous term
    x0 : np.array
        Initial guess of linear problem

    Returns
    --------
    x : np.array
        The solution

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
    V, H, g, _ = gmres_factorize(A, b, x0, krylov_tol=krylov_tol,
                                 krylov_maxiter=krylov_maxiter, **cfg)
    y = np.linalg.solve(H, g)
    return np.dot(V, y)
