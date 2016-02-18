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


def array_adapter(method):
    def wrapper(func, x, *args, **kwds):
        shape = x.shape
        dtype = x.dtype
        N = x.size

        def convert(y):
            y.reshape(N)
            if dtype is np.complex:
                y = np.concatenate((np.real(y), np.imag(y)))
            return y

        def revert(y):
            if dtype is np.complex:
                y = y[:len(y)/2] + 1j * y[len(y)/2:]
            y.reshape(shape)
            return y

        f = lambda y: convert(func(revert(y)))
        obj = method(f, convert(x), *args, **kwds)
        if isinstance(obj, tuple):
            return _apply_first(revert, obj)
        if isinstance(obj, types.GeneratorType):
            return _apply_first_gen(revert, obj)
        else:
            return revert(obj)
    return wrapper
