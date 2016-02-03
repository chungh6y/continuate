# -*- coding: utf-8 -*-

import numpy as np
from . import qr

from logging import getLogger, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)


class Arnoldi(object):

    def __init__(self, A, b, eps=1e-6, initialize=True):
        self.A = A
        self.ortho = qr.MGS()
        self.ortho(b)
        self.eps = eps
        self.coefs = []
        if initialize:
            self.calc()

    def __iter__(self):
        return self.ortho.__iter__()

    def __getitem__(self, i):
        return self.ortho[i]

    def basis(self):
        return self.__iter__()

    def calc(self):
        while True:
            Av = self.A * self.ortho[-1]
            coef = self.ortho(Av)
            logger.debug("Residual of Arnoldi iteration = {}".format(coef[-1]))
            self.coefs.append(coef)
            if coef[-1] < self.eps:
                return

    def projected_matrix(self):
        N = len(self.coefs)
        H = np.zeros((N, N))
        for i, c in enumerate(self.coefs):
            n = min(N, len(c))
            H[:n, i] = c[:n]
        return H


def arnoldi(A, b, **kwds):
    A = Arnoldi(A, b, **kwds)
    return A.projected_matrix(), np.stack(A)
