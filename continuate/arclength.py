# -*- coding: utf-8 -*-

""" numerical continuation with tangent space

Options
--------
tangentspace_dmu : float
    Infinitesimal of parameter :math:`d\mu` for calculating :math:`dx/d\mu`

"""

from . import newton, krylov
from .misc import Logger, array_adapter_p
import numpy as np
from itertools import count as icount

default_options = {
    "tangent_dmu": 1e-7,
}


def concat(x, mu):
    """ Convert :math:`(x, \mu)` to :math:`\\xi` """
    return np.concatenate((x, [mu]))


@array_adapter_p
def tangent_vector(func, x, mu, tangent_dmu=1e-7, dxi=None, **opt):
    """ Get tangent vector at :math:`(x, \mu)`

    Parameters
    -----------
    func : (numpy.array, float) -> numpy.array
        :math:`F(x, \mu)`,
        :code:`func(x, mu)` must have same dimension of :code:`x`

    Returns
    --------
    dxi : np.array
        Tangent vector
    """
    logger = Logger(__name__, "TangentVector")
    dmu = tangent_dmu
    dfdmu = (func(x, mu+dmu) - func(x, mu)) / dmu
    J = newton.Jacobi(lambda y: func(y, mu), x, **opt)
    dxdmu = krylov.gmres(J, -dfdmu, **opt)
    v = concat(dxdmu, 1)
    v /= np.linalg.norm(v)
    if dxi is None:
        return v[:-1], v[-1]
    vdxi = np.dot(dxi, v)
    logger.debug({"(v, dxi)": vdxi, })
    if vdxi < 0:
        v *= -1
    return v[:-1], v[-1]


@array_adapter_p
def continuation(func, x, mu, delta, **opt):
    """ Generator for continuation of a vector function :math:`F(x, \mu)`

    Using Newton-Krylov-Hook algorithm in each of continuation steps.

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

    Yields
    -------
    x : numpy.array
    mu : float

    """
    logger = Logger(__name__, "Continuation")
    xi = concat(x, mu)
    dxi = concat(np.zeros_like(x), delta)
    for t in icount():
        logger.info({"count": t, "mu": xi[-1], })
        yield xi[:-1], xi[-1]
        dxi = concat(*tangent_vector(func, xi[:-1], xi[-1], dxi=dxi, **opt))
        xi0 = xi + abs(delta) * dxi
        f = lambda z: concat(func(z[:-1], z[-1]), np.dot(z-xi0, dxi))
        xi = newton.newton_krylov_hook(f, xi, **opt)
        logger.debug({
            "count": t,
            "|f(x)|": np.linalg.norm(func(xi[:-1], xi[-1])),
            "dmu": abs(delta)*dxi[-1],
            "delta mu": xi[-1] - xi0[-1],
            "(dxi, xi-xi0)": np.dot(xi-xi0, dxi),
        })
