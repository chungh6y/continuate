# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse.linalg import LinearOperator

from logging import getLogger, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)


def Jacobian(func, x0, alpha=1e-7, fx=None):
    """
    Jacobi oprator :math:`DF(x0)`, where

    .. math::
        DF(x0)dx = (F(x0 + \\alpha dx / |dx|) - F(x0))/(\\alpha/|dx|)

    Parameters
    ----------
    func: function

    x0: numpy.array

    alpha: float

    fx: numpy.array, optional
        func(x0)

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x: np.array([x[1]**2, x[0]**2])
    >>> x0 = np.array([1, 2])
    >>> J = Jacobian(f, x0)

    """
    if fx is None:
        fx = func(x0)

    def wrap(v):
        norm = np.linalg.norm(v)
        r = alpha / norm
        return (func(x0 + r * v) - fx) / r

    return LinearOperator((len(x0), len(x0)), matvec=wrap, dtype=x0.dtype)
