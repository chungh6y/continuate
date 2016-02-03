# -*- coding: utf-8 -*-


import numpy as np


class MGS(object):
    """ modified Gram-Schmit

    >>> mgs = MGS()
    >>> for i in range(10):
    ...     u = np.random.random(9)
    ...     prod, norm = mgs(u)
    >>> len(mgs) == 9  # last insert does not add vector
    True
    >>> np.allclose(norm, 0)
    True
    >>> ortho = []
    >>> for i in range(len(mgs)):
    ...     for j in range(i):
    ...         ortho.append(np.dot(mgs[i], mgs[j]))
    >>> np.allclose(ortho, np.zeros_like(ortho))
    True
    >>> norms = [np.linalg.norm(u) for u in mgs]
    >>> np.allclose(norms, np.ones_like(norms))
    True

    """

    def __init__(self, eps=1e-9, dot=np.dot):
        self.v = []
        self.dot = dot
        self.e = eps

    def __iter__(self):
        return self.v.__iter__()

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        return self.v[i]

    def __call__(self, u):
        inner_prod = []
        for v in self.v:
            uv = self.dot(u, v)
            u -= uv * v
        u_norm = np.sqrt(self.dot(u, u))
        if u_norm > self.e:
            self.v.append(u / u_norm)
        return inner_prod, u_norm
