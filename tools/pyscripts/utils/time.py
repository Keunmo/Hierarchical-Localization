# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# utils.time
#
# Author: Changhee Won (changhee.won@multipleye.co)
#
#
import time
from datetime import datetime

def sec_to_date_str(sec: float, fmt: str = '%y%m%d_%H%M%S') -> str:
    date = datetime.fromtimestamp(sec)
    return date.strftime(fmt)

def ns_to_date_str(ns: int, fmt: str = '%y%m%d_%H%M%S') -> str:
    return sec_to_date_str(ns * 1e-9, fmt)

def timestamp_in_ns() -> int:
    return int(time.time() * 1e9)

def current_date_str(fmt: str = '%y%m%d_%H%M%S') -> str:
    return sec_to_date_str(time.time(), fmt)
