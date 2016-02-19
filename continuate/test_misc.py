# -*- coding: utf-8 -*-

import numpy as np
import unittest
from . import misc


@misc.array_adapter
def apply_func(func, x):
    return func(x)


def sample_gen():
    for i in range(10):
        yield i, i*i


class TestArrayAdapter(unittest.TestCase):

    def test_apply_first(self):
        f = lambda x: 2*x
        t = 2, 3, 4
        s = misc._apply(f, t)
        self.assertEqual(type(s), tuple)
        self.assertEqual(s, (4, 3, 4))

    def test_apply_first_gen(self):
        f = lambda x: 2*x
        G = misc._apply_first_gen(f, sample_gen())
        for t, (a, b) in enumerate(G):
            self.assertEqual(a, 2*t)
            self.assertEqual(b, t*t)

    def test_apply(self):
        f = lambda x: 2*x
        self.assertEqual(misc._apply(f, 2), 4)
        s = misc._apply(f, (2, 3, 4))
        self.assertEqual(type(s), tuple)
        self.assertEqual(s, (4, 3, 4))
        G = misc._apply(f, sample_gen())
        for t, (a, b) in enumerate(G):
            self.assertEqual(a, 2*t)
            self.assertEqual(b, t*t)

    def test_array_adapter(self):
        shape = (2, 3)

        def f(x):
            self.assertEqual(x.shape, shape)
            return 2*x
        x = np.ones(shape)
        ad = misc.ArrayAdapter(f, x)
        y = ad.convert(x)
        np.testing.assert_allclose(y, np.ones(6))
        y = ad(y)
        np.testing.assert_allclose(y, 2*np.ones(6))
