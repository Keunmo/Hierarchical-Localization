# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# capture.pixel_format.py
# convert pixel format
#
# Author: Changhee Won (changhee.won@multipleye.co)
#
#
from struct import unpack, pack
from utils.common import *

def yuv2red(Y, U, V): return np.clip((Y + 409 * V + 128) >> 8, 0, 255)
def yuv2green(Y, U, V): return np.clip((Y - 100 * U - 208 * V + 128) >> 8, 0, 255)
def yuv2blue(Y, U, V): return np.clip((Y + 516 * U + 128) >> 8, 0, 255)

def yuv411_to_rgb8_y4uv(src: np.ndarray, width: int, height: int) -> np.ndarray:
    Y1 = 298 * (src[::6].astype(np.int32) - 16).reshape((-1, 1))
    Y2 = 298 * (src[1::6].astype(np.int32) - 16).reshape((-1, 1))
    Y3 = 298 * (src[2::6].astype(np.int32) - 16).reshape((-1, 1))
    Y4 = 298 * (src[3::6].astype(np.int32) - 16).reshape((-1, 1))
    U = (src[4::6].astype(np.int32) - 128).reshape((-1, 1))
    V = (src[5::6].astype(np.int32) - 128).reshape((-1, 1))
    R1 = yuv2red(Y1, U, V); G1 = yuv2green(Y1, U, V); B1 = yuv2blue(Y1, U, V)
    R2 = yuv2red(Y2, U, V); G2 = yuv2green(Y2, U, V); B2 = yuv2blue(Y2, U, V)
    R3 = yuv2red(Y3, U, V); G3 = yuv2green(Y3, U, V); B3 = yuv2blue(Y3, U, V)
    R4 = yuv2red(Y4, U, V); G4 = yuv2green(Y4, U, V); B4 = yuv2blue(Y4, U, V)
    out = np.concatenate(
        (R1, G1, B1, R2, G2, B2, R3, G3, B3, R4, G4, B4), 1).reshape(
            (height, width, 3)).astype(np.uint8)
    return out
