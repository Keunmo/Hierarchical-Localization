# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# utils.file_io.py
#
# Author: Changhee Won (changhee.won@multipleye.co)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import *

from utils.common import *
from utils.array import *

def load_gps_csv_file(path: str, delimiter:Any = ',') \
        -> Union[Tuple[np.ndarray, np.ndarray], None]:
    '''
        return: (gps(lat, lng, alt), timestamps)
    '''
    if not osp.exists(path):
        LOG_ERROR('GPS csv file does not exist: ' + path)
        return None
    ifs = open(path, 'r')
    ifs.readline() # skip header
    data = np.loadtxt(ifs, delimiter=delimiter,
                      usecols=(0, 1, 2, 3)) # (ts, lat, lng, alt)
    timestamps = data[:, 0].astype(int).flatten()
    gps = data[:, 1:4].T.astype(float).reshape((3, -1))
    return (gps, timestamps)

def load_lla_text_file(path: str, delimiter:Any = ' ') \
        -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], None]:
    '''
        return: (gps(lat, lng, alt), timestamps, fidxs)
    '''
    if not osp.exists(path):
        LOG_ERROR('LLA txt file does not exist: ' + path)
        return None
    ifs = open(path, 'r')
    data = np.loadtxt(ifs, delimiter=delimiter)
    timestamps = data[:, 0].astype(int).flatten()
    gps = data[:, 1:4].T.astype(float).reshape((3, -1))
    fidxs = data[:, 4].astype(int).flatten()
    return (gps, timestamps, fidxs)

def load_pose_text_file(path: str, delimiter:Any = ' ') \
        -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], None]:
    '''
        return: (fidxs, poses, timestamps)
    '''
    if not osp.exists(path):
        LOG_ERROR('Pose file does not exist: ' + path)
        return None
    data = np.loadtxt(path, delimiter=delimiter)
    if data.ndim==1:
        data = data[None,:]
    fidxs = data[:, 0].astype(int).flatten()
    poses = data[:, 1:7].T.reshape((6, -1))
    timestamps = np.loadtxt(path, dtype=int, delimiter=' ', usecols=7).flatten()
    # timestamps = data[:, 7].astype(int).flatten()
    nposes = poses.shape[1]
    VLOG(1, '%d poses loaded', nposes)
    return (fidxs, poses, timestamps)

def load_imu_csv_file(path: str, delimiter:Any = ',') \
        -> Union[np.ndarray ,None]:
    '''
        return: (gyro, acc, timestamps)
    '''
    if not osp.exists(path):
        LOG_ERROR('Imu csv file does not exist: ' + path)
        return None
    ifs = open(path, 'r')
    ifs.readline() # skip desc
    ifs.readline() # skip header
    data = np.loadtxt(ifs, delimiter=delimiter)
    timestamps = data[:, 0].astype(int).flatten()
    gyro = data[:, 1:4].T.astype(float).reshape((3, -1))
    acc = data[:, 4:7].T.astype(float).reshape((3, -1))
    ndata = data.shape[1]
    return (gyro, acc, timestamps)


