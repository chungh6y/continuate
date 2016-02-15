# -*- coding: utf-8 -*-

import os.path as op
from glob import glob
from importlib import import_module


__all__ = [op.basename(f)[:-3]
           for f in glob(op.join(op.dirname(__file__), "*.py"))
           if op.basename(f) != "__init__.py"
           if not op.basename(f).startswith("test_")]

_mods = [import_module("continuate." + m) for m in __all__]


def get_default_options():
    """ get default options as a dict

    Examples
    ---------
    >>> get_default_options()
    {'krylov_maxiter': 1000.0, 'krylov_tol': 1e-09, 'newton_maxiter': 100, 'jacobi_alpha': 1e-07, 'newton_tol': 1e-07, 'hook_maxiter': 100, 'hook_tol': 0.1, 'trusted_region': 0.1}

    """
    cfg = {}
    for m in _mods:
        if hasattr(m, "default_options"):
            cfg.update(m.default_options)
    return cfg
