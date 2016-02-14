# -*- coding: utf-8 -*-

from . import newton, krylov
import numpy as np


class TangentSpace(object):
    """ Tangent space at :math:`(x, \mu)`

    Attributes
    -----------
    H : numpy.array
        Krylov-projected matrix of Jacobian :math:`DF(x, \mu)`
    V : numpy.array
        Basis yielded by Krylov subspace iteration,
        i.e. satisfies :math:`DF(x, \mu)V = VH`.
    tangent_vector : numpy.array
        normalized tangent vector :math:`(dx, d\mu)`, where
        :math:`dx/d\mu = -DF(x, \mu)^{-1}F(x, \mu)`.

    Parameters
    -----------
    func : (numpy.array, float) -> numpy.array
        :math:`F(x, \mu)`,
        :code:`func(x, mu)` must have same dimension of :code:`x`
    alpha : float, optional
        relative inf small: :math:`d\mu = \\alpha \mu`

    """
    def __init__(self, func, x, mu, alpha=1e-7, inner_tol=1e-9):
        dmu = mu * alpha
        dfdmu = (func(x, mu+dmu) - func(x, mu)) / dmu
        J = newton.Jacobi(lambda y: func(y, mu), x, alpha=alpha)
        self.H, self.V = krylov.arnoldi(J, dfdmu, eps=inner_tol)
        g = krylov.solve_Hessenberg(self.H, krylov.norm(dfdmu))
        dxdmu = np.dot(self.V[:, :len(g)], g)
        v = np.concatenate((dxdmu, [1]))
        self.tangent_vector = v / krylov.norm(v)

    def projected(self):
        """ Krylov-projected matrix and basis

        Returns
        --------
        (H, V)

        """
        return self.H, self.V


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
    J = newton.Jacobi(lambda y: func(y, mu), x, alpha=alpha)
    dx = krylov.gmres(J, -dfdmu)
    inv_norm = 1.0 / np.sqrt(np.dot(dx, dx) + 1)
    return inv_norm * dx, inv_norm


def continuate(func, x0, mu0, delta):
    """
    A generator for continuation

    Parameters
    -----------
    func: (numpy.array, float) -> numpy.array
        The function which will be continuated
    x0: numpy.array
        Initial point of continuation. It must satisfy `func(x0, mu0) = 0`
    mu0: float
        Initial parameter of continuation. It must satisfy `func(x0, mu0) = 0`
    delta: float
        step length of continuation

    Returns
    --------
    Genrator yielding (numpy.array, float)

    Examples
    ---------
    >>> import numpy as np
    >>> from itertools import islice
    >>> f = lambda x, mu: np.array([x[1]**2 - mu, x[0]])
    >>> x0 = np.array([1.0, 0.0])
    >>> mu0 = 1.0
    >>> G = continuate(f, x0, mu0, 1)
    >>> result = []
    >>> for x, m in islice(G, 10):
    ...     result.append((x, m))

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
        x1 = newton.newton(F, pre, ftol=1e-7)
        mu0 = mu(x1)
        x0 = x1
        yield x0, mu0
