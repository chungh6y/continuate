# -*- coding: utf-8 -*-

from . import newton, krylov
from .logger import Logger
import numpy as np
from itertools import count as icount


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
        normalized tangent vector :math:`d\\xi = (dx, d\mu)`, where
        :math:`dx/d\mu = -DF(x, \mu)^{-1}F(x, \mu)`.
        The sign is chosen as :math:`d\mu > 0`.

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
        self.H, self.V = krylov.arnoldi(J, -dfdmu, eps=inner_tol)
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


def continuation(func, x, mu, delta, alpha=1e-7, ftol=1e-7, inner_tol=1e-9):
    """ Generator for continuation of a vector function :math:`F(x, \mu)`

    Parameters
    -----------
    func : (numpy.array, float) -> numpy.array
        :math:`F(x, \mu)`
        :code:`func(x, mu)` must have same dimension of :code:`x`
    x : numpy.array
        Initial point of continuation, and satisfies :math:`F(x, \mu) = 0`
    mu : float
        Initial parameter of continuation, and satisfies :math:`F(x, \mu) = 0`
    delta : float
        step length of continuation.
        To decrease the parameter, you should set negative value.
    alpha : float, optional
        inf small of differentiation
    ftol : float, optional
        Stop criterion of Newton method
    inner_tol : float, optional
        Stop criterion of Krylov subspace generation

    Yields
    -------
    x : numpy.array
    mu : float
    ts : TangentSpace
        Tangent space at :math:`\\xi = (x, \mu)`

    """
    logger = Logger(__name__, "Continuation")
    concat = lambda x_, mu_: np.concatenate((x_, [mu_]))
    xi = concat(x, mu)
    dxi = concat(np.zeros_like(x), delta)
    for t in icount():
        logger.info({
            "count": t,
            "mu": xi[-1],
        })
        ts = TangentSpace(func, xi[:-1], xi[-1],
                          alpha=alpha, inner_tol=inner_tol)
        yield xi[:-1], xi[-1], ts
        if np.dot(dxi, ts.tangent_vector) < 0:
            dxi = -ts.tangent_vector
        else:
            dxi = ts.tangent_vector
        xi0 = xi + abs(delta) * dxi
        f = lambda z: concat(func(z[:-1], z[-1]), np.dot(z-xi0, dxi))
        xi = newton.newton(f, xi, ftol=ftol, inner_tol=inner_tol)
