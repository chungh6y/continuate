# -*- coding: utf-8 -*-

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
