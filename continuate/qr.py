# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm


def mgs(vectors, eps=1e-9):
    """ modified Gram-Schmit algorithm

    Parameters
    -----------
    vectors : iterator of np.arrays

    Examples
    ---------
    >>> from numpy.random import random
    >>> vecs = [random(5) for _ in range(10)]
    >>> qr = [np.dot(Q, r) for Q, r in mgs(vecs)]
    >>> np.testing.assert_allclose(qr, vecs)
    """
    Q = None
    for v in vectors:
        if Q is None:
            Q = np.zeros((0, len(v)))
        Qv = np.dot(Q, v)
        v = v - np.dot(Q.T, Qv)
        nv = norm(v)
        if nv > eps:
            Q = np.vstack((Q, v/nv))
            yield Q.T, np.append(Qv, nv)
        else:
            yield Q.T, Qv
