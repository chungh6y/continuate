# -*- coding: utf-8 -*-

from .linalg import Jacobian

import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp


def tangent_vector(func, x, mu, alpha=1e-7, dmu=None):
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
        alpha for Jacobian
    dmu: float, optional
        if None, alpha is used.

    Returns
    -------
    (dx, dmu): (numpy.array, float)
        normalized vector: :math:`dx^2 + dmu^2 = 1`

    """
    if dmu is None:
        dmu = alpha
    dfdmu = (func(x, mu + dmu) - func(x, mu)) / dmu
    J = Jacobian(lambda y: func(y, mu), x, alpha=alpha)
    dx, _ = sp.linalg.gmres(J, -dfdmu)
    inv_norm = 1.0 / np.sqrt(np.dot(dx, dx) + 1)
    return inv_norm * dx, inv_norm


def continuate(func, x0, mu0, delta):
    """
    A generator for continuation
    """
    while True:
        dx, dmu = tangent_vector(func, x0, mu0)
        mu = lambda x: mu0 + (delta - np.dot(x - x0, dx)) / dmu
        F = lambda x: func(x, mu(x))
        x = x0 + delta * dx
        x0 = opt.newton_krylov(F, x)
        mu0 = mu(x0)
        yield x0, mu0
