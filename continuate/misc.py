# -*- coding: utf-8 -*-

import numpy as np
import types
from logging import getLogger, DEBUG, LoggerAdapter


class Logger(LoggerAdapter):

    def __init__(self, module_name, algorithm_name):
        logger = getLogger(module_name)
        logger.setLevel(DEBUG)
        super(Logger, self).__init__(logger, {"algorithm": algorithm_name})

    def process(self, msg, kwds):
        if type(msg) is str:
            msg = {"message": msg}
        msg.update(self.extra)
        return msg, kwds


class ArrayAdapter(object):
    def __init__(self, func, x):
        self.func = func
        self.shape = x.shape
        self.dtype = x.dtype
        self.N = x.size

    def convert(self, y):
        y = y.reshape(self.N)
        if self.dtype == np.complex:
            y = np.concatenate((np.real(y), np.imag(y)))
        return y

    def revert(self, y):
        if self.dtype == np.complex:
            y = y[:len(y)/2] + 1j * y[len(y)/2:]
        y = y.reshape(self.shape)
        return y

    def __call__(self, x):
        return self.convert(self.func(self.revert(x)))


class ArrayAdapterP(ArrayAdapter):
    def __call__(self, x, mu):
        return self.convert(self.func(self.revert(x), mu))


def array_adapter(method):
    def wrapper(func, x, *args, **kwds):
        f = ArrayAdapter(func, x)
        v = f.convert(x)
        obj = method(f, v, *args, **kwds)
        return _apply(f.revert, obj)
    return wrapper


def array_adapter_p(method):
    def wrapper(func, x, *args, **kwds):
        f = ArrayAdapterP(func, x)
        obj = method(f, f.convert(x), *args, **kwds)
        return _apply(f.revert, obj)
    return wrapper


def _apply(func, obj):
    if isinstance(obj, tuple):
        return _apply_first(func, obj)
    if isinstance(obj, types.GeneratorType):
        return _apply_first_gen(func, obj)
    return func(obj)


def _apply_first(func, tpl):
    def gen():
        yield func(tpl[0])
        for o in tpl[1:]:
            yield o
    return tuple(gen())


def _apply_first_gen(func, gen):
    for t in gen:
        if isinstance(t, tuple):
            yield _apply_first(func, t)
        else:
            yield func(t)
