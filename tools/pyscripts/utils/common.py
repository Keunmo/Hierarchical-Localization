# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# utils.common.py
#
# Author: Changhee Won (changhee.won@multipleye.co)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# if sys.version_info.minor > 6:
# from __future__ import annotations
import sys
from typing import *
import os
import os.path as osp
import traceback
import time
import random
import copy
from enum import Enum, unique, auto

# preload frequently used modules (required)
from absl import app
from absl import flags; FLAGS = flags.FLAGS
from absl import logging
from absl.flags import DEFINE_flag, DEFINE_string, DEFINE_boolean, DEFINE_bool,\
    DEFINE_float, DEFINE_integer,  DEFINE_enum, DEFINE_enum_class, DEFINE_list,\
    DEFINE_spaceseplist, DEFINE_multi, DEFINE_multi_string, \
    DEFINE_multi_integer, DEFINE_multi_float, DEFINE_multi_enum, \
    DEFINE_multi_enum_class, DEFINE_alias
DEFINE_double = DEFINE_float
from utils.log import LOG_INFO, LOG_ERROR, LOG_WARNING, LOG_DEBUG, LOG_FATAL,\
    VLOG

import numpy as np
import numpy.random as npr
import scipy
import matplotlib
import matplotlib.pyplot as plt
from easydict import EasyDict as Edict

# optional modules
try:
    import torch
    TORCH_FOUND = True
except ImportError:
    TORCH_FOUND = False

try:
    import cv2
    OPENCV_FOUND = True
except ImportError:
    OPENCV_FOUND = False

# try:
#     import tensorflow as tf
#     TF_FOUND = True
# except ImportError:
#     TF_FOUND = False
EPS_HALF = 2**(-10)
EPS_SINGLE = 2**(-23)
EPS_DOUBLE = 2**(-52)
EPS = EPS_DOUBLE

@unique
class AutoEnum(Enum):
    def __new__(cls):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

def argparse(opts: Edict, varargin: Edict = None) -> Edict:
    if varargin is not None:
        for k in varargin:
            opts[k] = varargin[k]
    return opts

def random_seed(seed: int) -> None:
    np.random.seed(seed)
    if TORCH_FOUND:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

def random_index(n: int) -> np.ndarray:
    return (np.arange(n, dtype=int) + np.random.randint(n)) % n

def random_index_2x(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.randint(n) #
    index1x = (np.arange(n) + x) % n
    index2x = (np.arange(2 * n) + 2 * x) % (2*n)
    return (index1x, index2x)

def get_random_indices(n: int, index_sample_ratio=1) \
        -> Tuple[np.ndarray, np.ndarray]:
    if index_sample_ratio == 1:
        index = random_index(n)
        return (index, index)
    if n % index_sample_ratio != 0:
        LOG_ERROR('n should be divided by sample_ratio')
    n2 = n // index_sample_ratio
    r = index_sample_ratio
    x = np.random.randint(n2) #
    index_sample = (np.arange(n2, dtype=int) + x) % n2
    index = (np.arange(n2 * r, dtype=int) + r * x) % (r * n2)
    return (index, index_sample)

def get_random_roll(n: int, sample_ratio:int = 1) -> Tuple[int, int]:
    if n % sample_ratio != 0:
        LOG_ERROR('n should be divided by sample_ratio')
    n2 = n // sample_ratio
    r_sample = npr.randint(n2)
    r = r_sample * sample_ratio
    return (r, r_sample)

def is_torch_tensor(x) -> bool:
    return TORCH_FOUND and type(x) == torch.Tensor

def rand(min_v, max_v=None, shape=None) -> float:
    """ generate random values
    - rand(x): [0, x) or (x, 0] random value
    - rand(x, y), rand([x, y]): [x, y) random value
    - rand(x, y, [h, w]), rand([x, y], shape=[h, w]):
       [x, y) random values of [h, w] shape"""
    if max_v is None:
        v_range = np.array(min_v)
        if v_range.size == 2:
            return rand(v_range[0], v_range[1], shape)
        r = random.random() if shape is None else npr.random(shape)
        return r * min_v
    else:
        r = random.random() if shape is None else npr.random(shape)
        return r * (max_v - min_v) + min_v
    
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'