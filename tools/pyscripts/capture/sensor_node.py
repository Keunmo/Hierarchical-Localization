# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# capture.sensor_node.py
#
# Author: Changhee Won (changhee.won@multipleye.co)
#
#

from utils.common import *
from utils.image import *
from capture.sensor_data import *

def get_sensor_data_type(type_str: str) -> SensorDataType:
    if type_str == 'multi_camera' or type_str == 'camera_multi_files' or \
            type_str == 'camera_dc1394' or type_str == 'camera_spinnaker':
        return SensorDataType.CAMERA_FRAME
    elif type_str == 'imu_file' or type_str == 'imu_fsm' or \
            type_str == 'imu_xsens' or type_str == 'imu_eb' or \
            type_str == 'imu_wit':
        return SensorDataType.IMU
    elif type_str == 'lidar_file' or type_str == 'lidar_velodyne':
        return SensorDataType.LIDAR
    elif type_str == 'gps_file' or type_str == 'gps_m8n':
        return SensorDataType.GPS
    else:
        LOG_ERROR('failed to find sensor data type: ' + type_str)
        return SensorDataType.UNKNOWN

class CameraMultiFilesReader(SensorDataReader):
    def __init__(self, sensor_id: int):
        super(CameraMultiFilesReader, self).__init__(
            sensor_id, SensorDataType.CAMERA_FRAME)
        self.ncams: int = 0
        self.img_fmt: str = 'cam%d/%05d.png' # cam_idx, fidx
        self.config: Dict[str, Any] = {}
        self.text_data_list: List[str] = []
        self.dbpath: str = ''

    def open(self,
            config: Dict[str, Any], dbpath: str = '',
            timestamp_file: str = 'cam_timestamp.csv') -> bool:
        self.dbpath = dbpath
        self.img_fmt = config['img_fmt'] if 'img_fmt' in config.keys() else \
            config['filefmt']
        self.txt_filepath = osp.join(dbpath, timestamp_file)
        txt_file = open(self.txt_filepath, 'r')
        if txt_file.closed:
            LOG_ERROR('failed to open text file: ' + self.txt_filepath)
            return False
        if not 'capture_desc' in config.keys():
            LOG_ERROR('"capture_desc" is not defined in capture config')
            return False
        self.txt_file_lines = txt_file.readlines()
        self.header.capture_desc = config['capture_desc']
        self.header.field_names = self.txt_file_lines[0].strip().split(',')
        self.start_read_idx = int(self.txt_file_lines[1][1])
        self.end_read_idx = int(self.txt_file_lines[-1][1])
        self.ncams = len(config['sensor_nodes'])
        self.config = config
        return True

    def get_text_data(self, idx: int) -> List[str]:
        if self.read_index_list:
            idx = self.read_index_list[idx]
        return self.txt_file_lines[idx + 1].strip().split(',')

    def __read_sequential(self) -> Union[SensorData, None]:
        if self.read_index_list and \
           self.index_list_idx < len(self.read_index_list) - 1:
            self.step_read_idx = \
                self.read_index_list[self.index_list_idx + 1] - \
                self.read_index_list[self.index_list_idx]
            if self.step_read_idx <= 0:
                LOG_ERROR(
                    'SensorDataReader.read: read_index_list is not sorted')
            else:
                self.index_list_idx += 1

        if self.cur_read_idx >= len(self.txt_file_lines) - 1:
            LOG_ERROR('current record index is out of text data list')
            return None
        text_data = self.get_text_data(self.cur_read_idx)
        if not text_data:
            LOG_ERROR('invalid text data')
            return None
        data = CameraFrameData(self.sensor_id)
        data.timestamp_ns = int(text_data[0])
        data.fidx = int(text_data[1])
        imgs = self.__read_images(data.fidx)
        if imgs is None: return None
        data.set_meta_from_header(self.header.capture_desc)
        data.data = concat([x.flatten() for x in imgs], 0)
        if self.step_read_idx > 0:
            self.cur_read_idx += self.step_read_idx
        return data

    def __read_images(self, fidx: int) -> Union[List[np.ndarray], None]:
        imgs = []
        for i in range(self.ncams):
            path = osp.join(self.dbpath, self.img_fmt % (i + 1, fidx))
            I = read_image(path, read_or_die=False)
            if I is None:
                LOG_ERROR('failed to load image of frame [%05d]: %s' % (
                    fidx, path))
                return None
            imgs.append(I)
        return imgs

    def read(self, idx: Optional[int] = None) -> Union[SensorData, None]:
        if idx is None: return self.__read_sequential()
        text_data = self.get_text_data(idx)
        if not text_data:
            LOG_ERROR('invalid text data')
            return None
        data = CameraFrameData(self.sensor_id)
        data.timestamp_ns = int(text_data[0])
        data.fidx = int(text_data[1])
        imgs = self.__read_images(data.fidx)
        if imgs is None: return None
        data.set_meta_from_header(self.header.capture_desc)
        data.data = concat([x.flatten() for x in imgs], 0)
        return data

    def set_read_index_list(self, idxs: List[int]):
        if not idxs: return
        sorted_idxs = sorted(idxs)
        record_idx, index_idx = 0, 0
        self.read_index_list = []
        for text_data in self.txt_file_lines[1:]:
            text_data = text_data.strip().split(',')
            fidx = int(text_data[1])
            if fidx == sorted_idxs[index_idx]:
                self.read_index_list.append(record_idx)
                index_idx += 1
            record_idx += 1
            if index_idx >= len(sorted_idxs):
                break
        self.index_list_idx = 0
        self.start_read_idx = self.read_index_list[0]
        self.end_read_idx = self.read_index_list[-1]
        self.cur_read_idx = self.start_read_idx


def get_camera_data_reader_from_yaml(
        config: Dict[str, Any], capture_dir: str) \
            -> Union[SensorDataReader, None]:
    nodes = config['sensor_nodes']
    camera_node_name = ''
    for node in nodes:
        type_str = config[node]['type']
        if type_str == 'camera_multi_files' or type_str == 'multi_camera':
            data_type = type_str
            camera_node_name = node
            break
    if not camera_node_name:
        LOG_ERROR('camera node not found')
        return None
    cam_node_config = config[camera_node_name]
    if 'dataset' in config.keys():
        for k, v in config['dataset'].items():
            cam_node_config[k] = v
    if type_str == 'multi_camera':
        data_reader = SensorDataReader(0, SensorDataType.CAMERA_FRAME)
        filepath_prefix = osp.join(capture_dir, camera_node_name)
        if not data_reader.open(filepath_prefix, FLAGS.capture_bin_fmt):
            return None
    else:
        data_reader = CameraMultiFilesReader(0)
        if not data_reader.open(cam_node_config, capture_dir):
            return None
    return data_reader
