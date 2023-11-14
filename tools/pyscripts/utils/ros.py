# -*- coding: utf-8 -*-
#
# Copyright (c) 2022, MultiplEYE co. ltd.
# All rights reserved.
#
# utils.ros.py
#
# Author: Byeongheon Choi (byeongheon.choi@multipleye.co)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.common import *
from utils.array import *
from PIL import Image

import yaml

def world_to_ros_locations(world_locs: np.ndarray,
                        bg_config: Optional[Dict[str, any]] = None) \
        -> np.ndarray:
    ros_locs = world_locs[:, [2, 0, 1]]
    ros_locs[:, 1] = -ros_locs[:, 1]
    ros_locs[:, 2] = -ros_locs[:, 2]

    if bg_config is not None:
        res = bg_config['resolution']
        ros_locs[:, 0] /= res
        ros_locs[:, 1] /= res
        ros_locs[:, 0] -= (bg_config['origin'][0] / res)
        ros_locs[:, 1] -= (bg_config['origin'][1] / res)

    return ros_locs

def ros_to_world_locations(ros_locs: np.ndarray,
                        bg_config: Optional[Dict[str, any]] = None) \
        -> np.ndarray:
    if bg_config is not None:
        res = bg_config['resolution']
        ros_locs[:, 0] *= res
        ros_locs[:, 1] *= res
        ros_locs[:, 0] += bg_config['origin'][0]
        ros_locs[:, 1] += bg_config['origin'][1]

    world_locs = ros_locs
    world_locs[:, 1] = -world_locs[:, 1]
    if ros_locs.shape[1] >= 3:
        world_locs[:, 2] = -world_locs[:, 2]
        world_locs = world_locs[:, [1, 2, 0]]
    else:
        world_locs = np.c_[world_locs, np.zeros(ros_locs.shape[0])]
        world_locs = world_locs[:, [1, 2, 0]]

    return world_locs

def get_ros_2d_map(img_path: str = None, yaml_path: str = None) \
        -> Tuple[Image.Image, Dict[str, any]]:
    if img_path and len(img_path) > 0:
        img = Image.open(img_path)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if yaml_path and len(yaml_path):
            config = yaml.safe_load(open(yaml_path))

    return img, config
