# -*- coding: utf-8 -*-

import importlib

__all__ = ["linalg", "single_parameter"]

for m in __all__:
    importlib.import_module("continuate." + m)
