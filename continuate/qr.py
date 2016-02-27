# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm


class OrthonalityError(RuntimeError):
    pass


class MGS(object):
    """ modified Gram-Schmit

    Attributes
    -----------
    V : np.array(2d)
        Basis of the generated linear space. :code:`V.shape` is :math:`(n, N)`,
        where :math:`N` denote the dimension of the vector,
        and :math:`n` denotes the step number.

    Examples
    ---------
    >>> mgs = MGS()
    >>> for i in range(5):
    ...     u = np.random.random(9)
    ...     prod = mgs(u)
    >>> len(mgs)
    5
    >>> mgs.V.shape
    (5, 9)

    >>> for i in range(10):
    ...     u = np.random.random(9)
    ...     prod = mgs(u)
    >>> len(mgs)
    9
    >>> mgs.V.shape
    (9, 9)

    Raise :py:class:`OrthonalityError`
    if a new dimension cannot be created with `append` method.

    >>> mgs = MGS()
    >>> for i in range(10):
    ...     u = np.random.random(9)
    ...     prod = mgs.append(u)
    Traceback (most recent call last):
        ...
    continuate.qr.OrthonalityError: Linearly dependent

    """
    def __init__(self, eps=1e-9):
        self.V = None
        self.e = eps

    def __iter__(self):
        return self.V.__iter__()

    def __len__(self):
        if self.V is None:
            return 0
        return len(self.V)

    def __getitem__(self, i):
        if self.V is None:
            raise RuntimeError("Basis is not initialized")
        return self.V[i]

    def project(self, u):
        uv = np.dot(self.V, u)
        u -= np.dot(self.V.T, uv)
        return uv, u

    def append(self, u):
        return self(u, strict_mode=True)

    def __call__(self, u, strict_mode=False):
        if self.V is None:
            n_u = norm(u)
            self.V = u / n_u
            self.V = self.V.reshape((1, len(u)))
            return np.array([n_u])
        coef, res = self.project(u)
        r = norm(res)
        if r > self.e:
            self.V = np.vstack((self.V, res/r))
        elif strict_mode:
            raise OrthonalityError("Linearly dependent")
        return np.append(coef, r)
