# -*- coding: utf-8 -*-

import importlib
import os.path as op
from glob import glob

__all__ = [op.basename(f)[:-3]
           for f in glob(op.join(op.dirname(__file__), "*.py"))
           if op.basename(f) != "__init__.py"]

for m in __all__:
    importlib.import_module("continuate." + m)
