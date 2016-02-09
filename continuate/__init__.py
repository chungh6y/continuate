# -*- coding: utf-8 -*-

import importlib
import os.path as op
from glob import glob
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

__all__ = [op.basename(f)[:-3]
           for f in glob(op.join(op.dirname(__file__), "*.py"))
           if op.basename(f) != "__init__.py"]

for m in __all__:
    importlib.import_module("continuate." + m)
