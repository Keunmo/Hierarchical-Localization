# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# utils.log
#
# Author: Changhee Won (changhee.won@multipleye.co)
#
import os
from shutil import copyfile
from distutils.dir_util import copy_tree
from absl import logging

from utils.init import * # import MuliplEYE's modules
from utils.common import *
def _get_absl_log_prefix(record):
    """Returns the absl log prefix for the log record.

    Args:
        record: logging.LogRecord, the record to get prefix for.
    """
    created_tuple = logging.time.localtime(record.created)
    created_millisecond = int(record.created % 1.0 * 1e3)

    critical_prefix = ''
    level = record.levelno
    if logging._is_non_absl_fatal_record(record):
        # When the level is FATAL, but not logged from absl, lower the level so
        # it's treated as ERROR.
        level = logging.ERROR
        critical_prefix = logging._CRITICAL_PREFIX
    severity = logging.converter.get_initial_for_level(level)

    return '%c%02d%02d %02d:%02d:%02d.%03d %5d %s:%d] %s' % (
        severity,
        created_tuple.tm_mon,
        created_tuple.tm_mday,
        created_tuple.tm_hour,
        created_tuple.tm_min,
        created_tuple.tm_sec,
        created_millisecond,
        os.getpid(),
        record.filename,
        record.lineno,
        critical_prefix)

class _LogFormatter(logging.PythonFormatter):
    def format(self, record):
        prefix = _get_absl_log_prefix(record)
        return prefix + super(logging.PythonFormatter, self).format(record)

LOG_INFO = logging.info
LOG_ERROR = logging.error
LOG_WARNING = logging.warning
LOG_DEBUG = logging.debug
LOG_FATAL = logging.fatal
LOG_EXCEP = logging.exception
VLOG = logging.vlog
logging.get_absl_handler().setFormatter(_LogFormatter())

#=================================================================
# deprecated and replaced by absl-py's absl.logging
#=================================================================
# import sys
# import logging
# logging.basicConfig(
#     stream=sys.stderr,
#     level=logging.INFO,
#     format =
#         '[%(levelname).1s%(asctime)s.%(msecs)03d %(process)d ' \
#         '%(filename)s:%(lineno)d] %(message)s',
#     datefmt='%m%d %H:%M:%S')
# __logger = logging.getLogger()
# LOG_INFO = __logger.info
# LOG_ERROR = __logger.error
# LOG_WARNING = __logger.warning
# LOG_DEBUG = __logger.debug
# LOG_CRITICAL = __logger.critical

def file_backup(args):
    copy_tree("./",os.path.join(args.log_dir,'recording'))

# def file_backup(args):
#     dir_lis = ["./","./module","./python-scripts/utils","./utils","./python-scripts/capture"]
#     os.makedirs(os.path.join(args.log_dir, 'recording'), exist_ok=True)
#     for dir_name in dir_lis:
#         cur_dir = os.path.join(args.log_dir, 'recording', dir_name)
#         os.makedirs(cur_dir, exist_ok=True)
#         files = os.listdir(dir_name)
#         for f_name in files:
#             if f_name[-3:] == '.py' or f_name[-5:] == '.yaml':
#                 copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
