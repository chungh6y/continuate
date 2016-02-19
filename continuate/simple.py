# -*- coding: utf-8 -*-

""" Simple parameter continuation """

from . import newton
from .misc import Logger, array_adapter_p
from itertools import count as icount


@array_adapter_p
def continuation(func, x, mu, delta, **opt):
    """

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

    """
    logger = Logger(__name__, "Continuation")
    for t in icount():
        mu += delta
        f = lambda x: func(x, mu)
        x = newton.newton_krylov_hook(f, x, **opt)
        logger.info({
            "count": t,
            "mu": mu,
        })
        yield x, mu
