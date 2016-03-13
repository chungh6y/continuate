# -*- coding: utf-8 -*-

import numpy as np
from unittest import TestCase
from . import arclength, get_default_options


class TestTangentSpace(TestCase):
    N = 3
    opt = get_default_options()

    def _rand(self):
        return np.random.random(self.N)

    def _zeros(self):
        return np.zeros(self.N)

    def test_linear(self):
        """
        Linear space :math:`x(\mu) = \mu a`,
        where :math:`a` is constant vector
        """
        a = self._rand()
        a /= np.linalg.norm(a)
        f = lambda x, mu: x - mu*a
        x0 = np.zeros_like(a)
        dx, dmu = arclength.tangent_vector(f, x0, 0, **self.opt)
        dxi = arclength.concat(dx, dmu)
        np.testing.assert_allclose(dx/np.linalg.norm(dx), a)
        np.testing.assert_almost_equal(np.linalg.norm(dxi), 1)

    def test_parabola(self):
        """
        :math:`f(x, \mu) = x - \mu^2 a`,
        where :math:`a` is constant vector
        """
        a = self._rand()
        a /= np.linalg.norm(a)
        f = lambda x, mu: x - mu*mu*a

        dx, dmu = arclength.tangent_vector(f, 4*a, 2, **self.opt)
        dxi = arclength.concat(dx, dmu)
        np.testing.assert_allclose(dx/np.linalg.norm(dx), a)
        np.testing.assert_almost_equal(np.linalg.norm(dxi), 1)

        dx, dmu = arclength.tangent_vector(f, self._zeros(), 0, **self.opt)
        dxi = arclength.concat(dx, dmu)
        np.testing.assert_allclose(dx, self._zeros(), atol=1e-7)
        np.testing.assert_almost_equal(np.linalg.norm(dxi), 1)
