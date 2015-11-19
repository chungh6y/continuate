# -*- coding: utf-8 -*-

from .linalg.newton import Jacobi

import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp

from logging import getLogger, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)


def tangent_vector(func, x, mu, alpha=1e-7, alpha_mu=None):
    """
    Tangent vector at (x, mu)

    Parameters
    ----------
    func: (numpy.array, float) -> numpy.array

    x: numpy.array
        the position where the tangent space is calculated
    mu: float
        the paramter where the tangent space is calculated
    alpha: float
        alpha for Jacobi
    alpha_mu: float, optional
        if None, alpha is used.

    Returns
    -------
    (dx, dmu): (numpy.array, float)
        normalized vector: :math:`dx^2 + dmu^2 = 1`

    """
    if alpha_mu is None:
        alpha_mu = alpha
    dfdmu = (func(x, mu + alpha_mu) - func(x, mu)) / alpha_mu
    J = Jacobi(lambda y: func(y, mu), x, alpha=alpha)
    dx, _ = sp.linalg.gmres(J, -dfdmu)
    inv_norm = 1.0 / np.sqrt(np.dot(dx, dx) + 1)
    return inv_norm * dx, inv_norm


def continuate(func, x0, mu0, delta):
    """
    A generator for continuation
    """
    pre_dx = None
    while True:
        dx, dmu = tangent_vector(func, x0, mu0)
        # Keep continuation direction for overcoming turning points
        if pre_dx is not None and np.dot(pre_dx, dx) < 0:
            dx = -dx
            dmu = -dmu
        pre_dx = dx
        mu = lambda x: mu0 + (delta - np.dot(x - x0, dx)) / dmu
        F = lambda x: func(x, mu(x))
        pre = x0 + delta * dx
        x1 = opt.newton_krylov(F, pre, f_tol=1e-7)
        logger.debug(np.linalg.norm(F(x1)))
        mu0 = mu(x1)
        x0 = x1
        yield x0, mu0
