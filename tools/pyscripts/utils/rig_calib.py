# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# utils.rig_calib.py
#
# Author: Changhee Won (changhee.won@multipleye.co)
#
#
from utils.common import *
from utils.camera import *
from capture.sensor_data import *
from capture.sensor_node import *

class RigCalib:
    def __init__(self):
        self.cams_ = [] # List[CameraModel]
        self.rig2imu_poses_ = {} # Dict[np.ndarray]
        self.rig2lidar_poses_ = {} # Dict[np.ndarray]
        self.baseline_ = 0.0

    def get_ref_rig2imu_pose(self) -> np.ndarray:
        if not self.rig2imu_poses_: return np.eye(3, 4)
        return self.rig2imu_poses_[list(self.rig2imu_poses_.keys())[0]]

    @staticmethod
    def from_yaml_file(path: str) -> Union['RigCalib', None]: 
        '''
            Use if the capture.yaml has not been loaded. Automatically lead to the from_dict().
        '''
        if not osp.exists(path):
            LOG_ERROR('YAML file does not exist: ' + path)
            return None
        config = yaml.safe_load(open(path))
        yaml_dir = osp.split(path)[0]
        return RigCalib.from_dict(config, yaml_dir)

    @staticmethod
    def from_dict(
            config: Dict[str, any], yaml_dir: str = "") \
                -> Union['RigCalib', None]:
        if yaml_dir: config['_yaml_path'] = yaml_dir
        rig_calib = RigCalib()
        nodes = config['sensor_nodes']
        calib_splitted = 'calib' in config.keys()
        node_idx = 0
        for node in nodes:
            node_cfg = config[node]
            type_str = node_cfg['type']
            data_type = get_sensor_data_type(type_str)
            if data_type is SensorDataType.CAMERA_FRAME:
                # camera node
                if type_str == 'multi_camera' or \
                        type_str == 'camera_multi_files': # multi_camera
                    for cam_name in node_cfg['sensor_nodes']:
                        if calib_splitted:
                            calib = config['calib'][node][cam_name]
                        else:
                            if 'calib' in node_cfg.keys():
                                calib = node_cfg['calib'][cam_name]
                            elif 'calib' in node_cfg[cam_name].keys():
                                calib = node_cfg[cam_name]['calib']
                            else:
                                LOG_FATAL('calib node is not found')
                        rig_calib.cams.append(CameraModel.create_from_dict(calib))
                else: #single camera
                    if calib_splitted:
                        calib = config['calib'][node]
                    else:
                        calib = node_cfg['calib']
                    rig_calib.cams.append(CameraModel.create_from_dict(calib))
            elif data_type is SensorDataType.IMU:
                node_id = node_idx | data_type.value
                if 'imu2rig_pose' in node_cfg.keys():
                    imu2rig_vec = np.array(
                        node_cfg['imu2rig_pose']).reshape(6, 1)
                    rig_calib.rig2imu_poses[node_id] = inverse_transform(
                        to_pose_matrix(imu2rig_vec))
            elif data_type is SensorDataType.LIDAR:
                node_id = node_idx | data_type.value
                if 'lidar2rig_pose' in node_cfg.keys():
                    lidar2rig_vec = np.array(
                        node_cfg['lidar2rig_pose']).reshape(6, 1)
                    rig_calib.rig2lidar_poses[node_id] = inverse_transform(
                        to_pose_matrix(lidar2rig_vec))
            node_idx += 1
        # apply invalid_mask & compute baseline
        morph_filter = np.ones((5, 5), dtype=np.uint8)
        for cam in rig_calib.cams:
            rig_calib.baseline += norm(get_tr(cam.cam2rig))
            invalid_mask = None
            if cam.invalid_mask_file is not None:
                mask_file = osp.join(yaml_dir, cam.invalid_mask_file)
                if osp.exists(mask_file):
                    invalid_mask = read_image(mask_file).astype(np.bool)
                    if len(invalid_mask.shape) > 2:
                        invalid_mask = invalid_mask[..., 0] > 0
                        write_image(
                            invalid_mask.astype(np.uint8) * 255, mask_file)
                else:
                    LOG_WARNING(
                        'invalid mask file does not exist: ' + mask_file)
            cam.invalid_mask = cam.make_fov_mask(invalid_mask)
            cam.invalid_mask = ndimage.binary_closing(
                cam.invalid_mask, morph_filter, border_value=1)
        rig_calib.baseline /= len(rig_calib.cams)
        return rig_calib

    @property
    def cams(self) -> List[CameraModel]: return self.cams_
    
    @cams.setter
    def cams(self, v: List[CameraModel]): self.cams_ = v
    
    @property
    def rig2imu_poses(self) -> Dict[int, np.ndarray]: return self.rig2imu_poses_

    @rig2imu_poses.setter
    def rig2imu_poses(self, v: Dict[int, np.ndarray]): self.rig2imu_poses_ = v

    @property
    def rig2lidar_poses(self) -> Dict[int, np.ndarray]:
        return self.rig2lidar_poses_

    @rig2lidar_poses.setter
    def rig2lidar_poses(self, v: Dict[int, np.ndarray]):
        self.rig2lidar_poses_ = v

    @property
    def baseline(self) -> float: return self.baseline_
    
    @baseline.setter
    def baseline(self, v: float): self.baseline_ = v
