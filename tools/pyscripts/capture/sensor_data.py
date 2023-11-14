# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# capture.sensor_data.py
#
# Author: Changhee Won (changhee.won@multipleye.co)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from io import TextIOWrapper
from typing import *
from struct import unpack, pack
from enum import Enum, unique

from utils.common import *
from capture.pixel_format import *

DEFINE_string('capture_bin_fmt', '%03d.bin', 'capture binary file format')

@unique
class SensorDataType(Enum):
    UNKNOWN = 0
    CAMERA_FRAME = 0x0100
    IMU = 0x0200
    LIDAR = 0x0400
    GPS = 0x0800

class SensorDataHeader:
    def __init__(self):
        self.capture_desc = ''
        self.field_names = []

class SensorData:
    def __init__(self, sensor_id: int):
        self.sensor_id = sensor_id
        self.timestamp_ns = 0
        self.data = np.array([])

    def get_header(self) -> SensorDataHeader:
        raise NotImplementedError
    def get_text_data(self) -> List[str]:
        raise NotImplementedError
    def set_meta_from_header(self, capture_desc: List[str]) -> bool:
        raise NotImplementedError
    def set_meta_from_data(self, data: 'SensorData') -> bool:
        raise NotImplementedError
    def set_from_text_data(self, text_data: List[str]) -> bool:
        raise NotImplementedError

    @staticmethod
    def get_type(sensor_id: int) -> int:
        return sensor_id & 0xff00

    @staticmethod
    def get_type_name(sensor_id: int) -> str:
        t = SensorData.get_type(sensor_id)
        if (t == SensorDataType.UNKNOWN.value): return 'unknown'
        elif (t == SensorDataType.CAMERA_FRAME.value): return 'cam'
        elif (t == SensorDataType.IMU.value): return 'imu'
        elif (t == SensorDataType.LIDAR.value): return 'lidar'
        elif (t == SensorDataType.GPS.value): return 'gps'
        return ''

    def type(self):
        return SensorData.get_type(self.sensor_id)

    @property
    def binary_data(self) -> np.ndarray: return self.data

class CameraFrameData(SensorData):
    DATA_TYPE = SensorDataType.CAMERA_FRAME
    @unique
    class PixelFormat(AutoEnum):
        UNKNOWN = ()
        GRAY8 = ()
        BAYERRG8 = (); BAYERGR8 = (); BAYERGB8 = (); BAYERBG8 = ()
        GRAY16 = ()
        RGB8 = ()
        YUV411 = (); YUV411_UYYVYY = (); YUV411_YYUYYV = ()
        YUV422 = ()

    @unique
    class FlipType(AutoEnum):
        NONE = ()
        XFLIP = (); YFLIP = (); XYFLIP = (); XY_TRANSPOSE = ()

    @staticmethod
    def get_pixel_format_name(f: 'CameraFrameData.PixelFormat') -> str:
        if (f is CameraFrameData.PixelFormat.GRAY8): return 'gray8'
        elif (f is CameraFrameData.PixelFormat.BAYERBG8): return 'bayerRG8'
        elif (f is CameraFrameData.PixelFormat.BAYERGR8): return 'bayerGR8'
        elif (f is CameraFrameData.PixelFormat.BAYERGB8): return 'bayerGB8'
        elif (f is CameraFrameData.PixelFormat.BAYERBG8): return 'bayerBG8'
        elif (f is CameraFrameData.PixelFormat.GRAY16): return 'gray16'
        elif (f is CameraFrameData.PixelFormat.RGB8): return 'rgb8'
        elif (f is CameraFrameData.PixelFormat.YUV411): return 'yuv411'
        elif (f is CameraFrameData.PixelFormat.YUV411_UYYVYY):
            return 'yuv411_uyyvyy'
        elif (f is CameraFrameData.PixelFormat.YUV411_YYUYYV):
            return 'yuv411_yyuyyv'
        elif (f is CameraFrameData.PixelFormat.YUV411): return 'yuv422'
        return ''

    @staticmethod
    def get_pixel_format(f_str: str) -> 'CameraFrameData.PixelFormat':
        if (f_str == 'gray8') : return CameraFrameData.PixelFormat.GRAY8
        elif (f_str == 'bayerRG8') : return CameraFrameData.PixelFormat.BAYERRG8
        elif (f_str == 'bayerGR8') : return CameraFrameData.PixelFormat.BAYERGR8
        elif (f_str == 'bayerGB8') : return CameraFrameData.PixelFormat.BAYERGB8
        elif (f_str == 'bayerBG8') : return CameraFrameData.PixelFormat.BAYERBG8
        elif (f_str == 'gray16') : return CameraFrameData.PixelFormat.GRAY16
        elif (f_str == 'rgb8') : return CameraFrameData.PixelFormat.RGB8
        elif (f_str == 'yuv411') : return CameraFrameData.PixelFormat.YUV411
        elif (f_str == 'yuv411_uyyvyy') :
            return CameraFrameData.PixelFormat.YUV411_UYYVYY
        elif (f_str == 'yuv411_yyuyyv') :
            return CameraFrameData.PixelFormat.YUV411_YYUYYV
        elif (f_str == 'yuv422'): return CameraFrameData.PixelFormat.YUV422
        return CameraFrameData.PixelFormat.UNKNOWN

    @staticmethod
    def get_flip_type_name(f: 'CameraFrameData.FlipType') -> str:
        if (f is CameraFrameData.FlipType.NONE): return 'none'
        elif (f is CameraFrameData.FlipType.XFLIP): return 'x_flipped'
        elif (f is CameraFrameData.FlipType.YFLIP): return 'y_flipped'
        elif (f is CameraFrameData.FlipType.XYFLIP): return 'xy_flipped'
        elif (f is CameraFrameData.FlipType.XY_TRANSPOSE):
            return 'xy_transposed'
        return ''

    @staticmethod
    def get_flip_type(f_str: str) -> 'CameraFrameData.FlipType':
        if (f_str == 'none'): return CameraFrameData.FlipType.NONE
        elif (f_str == 'x_flipped'): return CameraFrameData.FlipType.XFLIP
        elif (f_str == 'y_flipped'): return CameraFrameData.FlipType.YFLIP
        elif (f_str == 'xy_flipped'): return CameraFrameData.FlipType.XYFLIP
        elif (f_str == 'xy_transposed'):
            return CameraFrameData.FlipType.XY_TRANSPOSE
        return CameraFrameData.FlipType.NONE

    @staticmethod
    def bits_per_pixel(f: 'CameraFrameData.PixelFormat') -> int:
        if (f is f.RGB8): return 24
        elif (f.value >= f.YUV422.value): return 16
        elif (f.value >= f.YUV411.value): return 12
        elif (f.value >= f.GRAY16.value): return 16
        return 8

    class Meta:
        def __init__(self):
            self.width, self.height = 0, 0
            self.pixfmt = CameraFrameData.PixelFormat.RGB8
            self.data_idx = 0
            self.flip = CameraFrameData.FlipType.NONE

        def get_frame_bytes(self) -> int:
            return int(self.width * self.height * \
                CameraFrameData.bits_per_pixel(self.pixfmt) / 8.0)

        def xy_transposed(self) -> bool:
            return (self.flip is CameraFrameData.FlipType.XY_TRANSPOSE)

    def __init__(self, sensor_id):
        super().__init__(sensor_id)
        self.meta = [] # List[Meta]
        self.fidx = -1

    def get_header(self) -> SensorDataHeader:
        header = SensorDataHeader()
        header.capture_desc = "# camera_frame %d" % (len(self.meta))
        for m in self.meta:
            pixfmt_str = CameraFrameData.get_pixel_format_name(m.pixfmt)
            if not pixfmt_str:
                raise ValueError('invalid pixel format')
            header.capture_desc += ' %d %d %s' % (m.width, m.height, pixfmt_str)
            if (m.flip is not CameraFrameData.FlipType.NONE):
                header.capture_desc += " %s" % (
                    CameraFrameData.get_flip_type_name(m.flip))
        header.field_names ['Timestamp (ns), fidx']
        return header

    def get_text_data(self) -> List[str]:
        return [str(self.timestamp_ns), str(self.fidx)]

    def set_meta_from_header(self, capture_desc: str) -> bool:
        try:
            tokens = capture_desc.split(' ')
            if tokens[0] != "#":
                raise ValueError('invalid format: ' + capture_desc)
            if tokens[1] != "camera_frame":
                raise ValueError('invalid format: ' + capture_desc)
            num_cameras = int(tokens[2])
            if num_cameras <= 0:
                raise ValueError('invalid num cameras: %d' % (num_cameras))
            self.meta = [CameraFrameData.Meta() for _ in range(num_cameras)]
            j = 3
            data_idx = 0
            for i, m in enumerate(self.meta):
                m.width = int(tokens[j])
                m.height = int(tokens[j + 1])
                pixfmt = tokens[j + 2]
                if pixfmt == "rgb8":
                    m.pixfmt = CameraFrameData.PixelFormat.RGB8
                elif pixfmt == "gray16":
                    m.pixfmt = CameraFrameData.PixelFormat.GRAY16
                elif pixfmt == "gray8":
                    m.pixfmt = CameraFrameData.PixelFormat.GRAY8
                elif pixfmt == "bayerRG8":
                    m.pixfmt = CameraFrameData.PixelFormat.BAYERRG8
                elif pixfmt == "bayerGB8":
                    m.pixfmt = CameraFrameData.PixelFormat.BAYERGB8
                elif pixfmt == "bayerGR8":
                    m.pixfmt = CameraFrameData.PixelFormat.BAYERGR8
                elif pixfmt == "bayerBG8":
                    m.pixfmt = CameraFrameData.PixelFormat.BAYERBG8
                elif pixfmt == "yuv411":
                    m.pixfmt = CameraFrameData.PixelFormat.YUV411
                else:
                    m.pixfmt = CameraFrameData.PixelFormat.UNKNOWN
                if m.width <= 0 or m.height <= 0 or \
                   m.pixfmt == CameraFrameData.PixelFormat.UNKNOWN:
                    raise ValueError(
                        'invalid meta %s x %s : %s for cam %d' % (
                            tokens[j], tokens[j + 1], tokens[j + 2], i))
                m.data_idx = data_idx
                data_idx += m.get_frame_bytes()
                m.flip = False
                if (len(tokens) >= j + 4):
                    flip = tokens[j + 3]
                    m.flip = CameraFrameData.get_flip_type(flip)
                    if (m.flip is not CameraFrameData.FlipType.NONE): j+= 1
                j += 3
        except ValueError as e:
            LOG_ERROR(e.args[0])
            return False
        return True

    def set_meta_from_data(self, data: SensorData) -> None:
        self.meta = CameraFrameData(data).meta.copy()

    def set_from_text_data(self, text_data: List[str]) -> bool:
        if len(text_data) < 1: return False
        self.timestamp_ns = int(text_data[0])
        if len(text_data) > 1: self.fidx = int(text_data[1])
        return True

    def has_color(self, idx: int) -> bool:
        m = self.meta[idx]
        if m.pixfmt is CameraFrameData.PixelFormat.GRAY8 or \
                m.pixfmt is CameraFrameData.PixelFormat.GRAY16:
            return False
        return True

    def has_color_all(self) -> bool:
        for i in range(len(self.meta)):
            if not self.has_color(i): return False
        return True

    def get_image(self, idx: int) -> Union[np.ndarray, None]:
        def check_opencv_and_print_err(f: CameraFrameData.PixelFormat):
            if not OPENCV_FOUND:
                LOG_ERROR('Converting %s needs OpenCV but not found' %
                    (CameraFrameData.get_pixel_format_name(f)))
                return False
            return True
        if idx < 0 or idx > len(self.meta): return None
        meta = self.meta[idx]
        out = self.data[meta.data_idx : meta.data_idx + meta.get_frame_bytes()]
        if meta.pixfmt == CameraFrameData.PixelFormat.RGB8:
            out = out.reshape((meta.height, meta.width, 3))
        elif meta.pixfmt == CameraFrameData.PixelFormat.BAYERRG8:
            out = out.reshape((meta.height, meta.width))
            if check_opencv_and_print_err(meta.pixfmt):
                out = cv2.cvtColor(out, cv2.COLOR_BAYER_RG2BGR)
        elif meta.pixfmt == CameraFrameData.PixelFormat.BAYERGB8:
            out = out.reshape((meta.height, meta.width))
            if check_opencv_and_print_err(meta.pixfmt):
                out = cv2.cvtColor(out, cv2.COLOR_BAYER_GB2BGR)
        elif meta.pixfmt == CameraFrameData.PixelFormat.BAYERGR8:
            out = out.reshape((meta.height, meta.width))
            if check_opencv_and_print_err(meta.pixfmt):
                out = cv2.cvtColor(out, cv2.COLOR_BAYER_GR2BGR)
        elif meta.pixfmt == CameraFrameData.PixelFormat.BAYERBG8:
            out = out.reshape((meta.height, meta.width))
            if check_opencv_and_print_err(meta.pixfmt):
                out = cv2.cvtColor(out, cv2.COLOR_BAYER_BG2BGR)
        elif meta.pixfmt == CameraFrameData.PixelFormat.YUV411:
            out = yuv411_to_rgb8_y4uv(out, meta.width, meta.height)
        elif meta.pixfmt == CameraFrameData.PixelFormat.GRAY16:
            out = out.view(np.uint16)
            out = out.reshape((meta.height, meta.width))
        else: # gray8
            out = out.reshape((meta.height, meta.width))
        if (meta.flip == CameraFrameData.FlipType.YFLIP):
            out = out[-1::-1, ...]
        return out

    def get_all_images(self) -> Union[List[np.ndarray], None]:
        if not self.meta: return None
        return [self.get_image(i) for i in range(len(self.meta))]

class LidarData(SensorData):
    DATA_TYPE = SensorDataType.LIDAR
    class Format(Enum): SEQ = 0; LOOP = 1

    def __init__(self, sensor_id):
        super().__init__(sensor_id)
        self.num_points = 0
        self.num_blocks = 0
        self.num_channels = 0

    def get_header(self) -> SensorDataHeader:
        header = SensorDataHeader()
        header.capture_desc = ""
        header.field_names ['Timestamp (ns)', 'Num pts', 'Num blocks']
        return header

    def get_text_data(self) -> List[str]:
        return [str(self.timestamp_ns),
            str(self.num_points), str(self.num_blocks)]

    def set_meta_from_header(self, capture_desc: str) -> bool:
        return True

    def set_meta_from_data(self, data: SensorData) -> None:
        data = LidarData(data)
        self.num_points = data.num_points
        self.num_blocks = data.num_channels
        self.num_channels = data.num_channels
        return True

    def set_from_text_data(self, text_data: List[str]) -> bool:
        if len(text_data) != 3: return False
        self.timestamp_ns = int(text_data[0])
        self.num_points = int(text_data[1])
        self.num_blocks = int(text_data[2])
        return True

    def points_idx(self): return 0
    def intensities_idx(self): return 3 * self.num_points * 4
    def azimuth_idxs_idx(self): return self.intensities_idx() + self.num_points
    def azimuth_degs_idx(self): return self.azimuth_idxs_idx() + self.num_points

    def points(self) -> np.ndarray:
        point_bytes = 3 * self.num_points * 4
        data = self.data[self.points_idx():]
        out = data[:point_bytes]
        out = out.view(dtype=np.float32)
        out = out.reshape((self.num_points, 3))
        return out

    def intensities(self) -> np.ndarray:
        data = self.data[self.intensities_idx():]
        out = data[:self.num_points]
        return out

    def azimuth_idxs(self) -> np.ndarray:
        data = self.data[self.azimuth_idxs_idx():]
        out = data[:self.num_points]
        return out

    def azimuth_degs(self) -> np.ndarray:
        data = self.data[self.azimuth_degs_idx():]
        out = data[:4 * self.num_blocks]
        out = out.view(dtype=np.float32)
        out = out.reshape((self.num_blocks))
        return out

    # (CVLidar is used in calib-matlab, will be deprecated)
    @staticmethod
    def write_lidar_list_as_cvlidar(
            data: Union[List['LidarData'], 'LidarData'],
            path: str,) -> bool:
        if type(data) != list:
            data = [data]
        num_seqs = len(data)
        with open(path, 'wb') as f:
            f.write(pack('i', num_seqs))
            for d in data:
                f.write(pack('i', d.num_points))
                d.points().tofile(f)
                d.intensities().tofile(f)
                d.azimuth_idxs().tofile(f)
                f.write(pack('i', d.num_blocks))
                d.azimuth_degs().tofile(f)
                f.write(pack('B', d.num_channels))
                f.write(pack('q', d.timestamp_ns))
        return True

class SensorDataReader:
    def __init__(self, sensor_id: int,
                 data_type: SensorDataType = SensorDataType.UNKNOWN,
                 use_multi_processing: bool = False):
        self.sensor_id: int = sensor_id | data_type.value
        self.header: SensorDataHeader = SensorDataHeader()
        self.read_index_list: List[int] = []
        self.start_read_idx: int = 0
        self.step_read_idx: int = 1
        self.end_read_idx: int = -1
        self.filepath_prefix: str = ''
        self.capture_bin_fmt: str = ''
        self.txt_filepath: str = ''
        self.txt_file_lines: List[str] = []
        self.bin_file: Union['BufferedReader', None] = None
        self.num_packets_per_file = 0
        self.use_multi_processing = use_multi_processing

    def open(self,
            filepath_prefix: str,
            capture_bin_fmt: str = '%03d.bin') -> bool:
        try:
            self.filepath_prefix = filepath_prefix
            self.capture_bin_fmt = capture_bin_fmt
            # text file
            self.txt_filepath = filepath_prefix + ".csv"
            txt_file = open(self.txt_filepath, 'r')
            if txt_file.closed:
                LOG_ERROR(
                    'SensorDataReader::Open: failed to open a text file %s' % (
                        self.txt_filepath))
                return False
            self.txt_file_lines = txt_file.readlines()
            txt_file.close()
            self.end_read_idx = len(self.txt_file_lines) - 3
            self.header.capture_desc = self.txt_file_lines[0].strip()
            self.header.field_names = self.txt_file_lines[1].strip().split(',')
            self.cur_read_idx = self.start_read_idx

            # binary file
            bin_filepath = filepath_prefix + capture_bin_fmt % 0
            bin_file = open(bin_filepath, 'rb')
            self.packet_size = unpack('i', bin_file.read(4))[0]
            bin_file.seek(0, 2)
            self.num_packets_per_file = bin_file.tell() // \
                (self.packet_size + 4)
            bin_file.close()
            self.bin_file_idx = self.start_read_idx // self.num_packets_per_file
            self.data_count = 0
            if self.read_index_list: self.index_list_idx = 0
            if self.use_multi_processing: return True
            bin_filepath = filepath_prefix + capture_bin_fmt % self.bin_file_idx
            self.bin_file = open(bin_filepath, 'rb')
            self.seek_binary_record(self.start_read_idx)
            return True
        except FileNotFoundError:
            LOG_ERROR('SensorDataReader::open: failed to open capture file')
            return False

    def seek_binary_record(self, record_idx: int) -> bool:
        if self.num_packets_per_file <= 0 or self.bin_file is None: return False
        file_idx = record_idx // self.num_packets_per_file
        if file_idx != self.bin_file_idx:
            self.bin_file.close()
            self.bin_file_idx = file_idx
            bin_filepath = self.filepath_prefix + \
                self.capture_bin_fmt % self.bin_file_idx
            try:
                self.bin_file = open(bin_filepath, 'rb')
            except:
                return False
        record_idx_in_file = record_idx - file_idx * self.num_packets_per_file
        if record_idx_in_file > 0:
            pos = record_idx_in_file * (self.packet_size + 4)
            self.bin_file.seek(pos)
        return True


    def set_read_index_range(self, start: int, end: int, step: int ) -> None:
        def get_fidx_from_line_if_exists(line: str, fidx: int):
            item = line.strip().split(',')
            if len(item) >= 2: return int(item[1])
            else: return fidx
        sensor_type = SensorData.get_type(self.sensor_id)
        if sensor_type != SensorDataType.CAMERA_FRAME.value:
            LOG_WARNING(
                'cannot set read index range to sensor [%d]', self.sensor_id)
            return
        if self.read_index_list:
            LOG_WARNING('SensorDataReader::setReadIndex: clear read_index_list')
            self.read_index_list = []
        start_read_idx = start
        self.step_read_idx = step
        txt_file_len = len(self.txt_file_lines)
        end_fidx = get_fidx_from_line_if_exists(
            self.txt_file_lines[-1], txt_file_len - 3)
        if end >= 0:
            end_read_idx = min(end, end_fidx)
        else:
            end_read_idx = end_fidx
        self.set_read_index_list(list(range(
            start_read_idx, end_read_idx + self.step_read_idx,
            self.step_read_idx)))

    def set_read_index_list(self, idxs: List[int]):
        def get_fidx_from_line_if_exists(line: str, fidx: int):
            item = line.strip().split(',')
            if len(item) >= 2: return int(item[1])
            else: return fidx
        if not idxs: return
        sensor_type = SensorData.get_type(self.sensor_id)
        if sensor_type != SensorDataType.CAMERA_FRAME.value:
            LOG_WARNING(
                'cannot set read index list to sensor [%d]', self.sensor_id)
            return
        sorted_idxs = sorted(idxs)
        self.read_index_list = []
        # re-open files
        if self.bin_file is not None and not self.bin_file.closed:
            self.bin_file.close()
        if not self.txt_file_lines:
            txt_file = open(self.txt_filepath, 'r')
            self.txt_file_lines = txt_file.readlines()
        header = self.txt_file_lines[1].strip().split(',')
        all_fidxs = np.array(
            [get_fidx_from_line_if_exists(
                self.txt_file_lines[i], i - 2) for i in range(
                    2, len(self.txt_file_lines))],
            dtype=int)
        is_kfidx = (all_fidxs[-1] - all_fidxs[0]) > 2 * (len(all_fidxs))
        if len(header) < 2: # does not have fidx, suppose ridx = fidx
            self.read_index_list = sorted_idxs
        else:
            if is_kfidx:
                start_ridx = np.searchsorted(all_fidxs, sorted_idxs[0])
                end_ridx = np.searchsorted(all_fidxs, sorted_idxs[-1])
                self.read_index_list = list(
                    range(start_ridx, end_ridx + self.step_read_idx,
                          self.step_read_idx))
            else:
                record_idx, index_idx = 0, 0
                for fidx in all_fidxs:
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
        self.bin_file_idx = self.start_read_idx // self.num_packets_per_file
        if self.use_multi_processing: return
        bin_filepath = self.filepath_prefix + self.capture_bin_fmt % (
            self.bin_file_idx)
        self.bin_file = open(bin_filepath, 'rb')
        self.seek_binary_record(self.start_read_idx)

    def get_text_data(self, idx: int) -> List[str]:
        return self.txt_file_lines[idx + 2].strip().split(',')

    def __read_sequential(self) -> Union[SensorData, None]:
        if self.use_multi_processing:
            LOG_ERROR(
                'SensorDataReader.read: cannot read sequentaily when'
                ' multi processing is on')
            return None
        if (self.end_read_idx >= 0 and self.cur_read_idx > self.end_read_idx):
            return None
        sensor_type = SensorData.get_type(self.sensor_id)
        data = SensorData(self.sensor_id)
        if sensor_type == SensorDataType.CAMERA_FRAME.value:
            data = CameraFrameData(self.sensor_id)
        elif sensor_type == SensorDataType.LIDAR.value:
            data = LidarData(self.sensor_id)
        else:
            LOG_ERROR('SensorDataReader.read: unknown type for %d' %
                (self.sensor_id))
            return None

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

        if self.cur_read_idx <= len(self.txt_file_lines) - 3:
            text_data = self.get_text_data(self.cur_read_idx)
            if sensor_type == SensorDataType.CAMERA_FRAME.value:
                data.fidx = self.cur_read_idx
            data.set_from_text_data(text_data)

        if not self.bin_file.closed:
            b = self.bin_file.read(4)
            if len(b) != 4:
                LOG_ERROR('SensorDataReader.read: failed to read bin file')
                self.close()
                return None
            packet_size = unpack('i', b)[0]
            data.data = np.fromfile(self.bin_file,
                dtype=np.uint8, count=packet_size)
            data.set_meta_from_header(self.header.capture_desc)
            self.data_count += 1

            if self.step_read_idx <= 0: return data
            self.cur_read_idx += self.step_read_idx
            self.seek_binary_record(self.cur_read_idx)
        return data

    def read(self, idx: Optional[int] = None) -> Union[SensorData, None]:
        if idx is None: return self.__read_sequential()
        if (self.end_read_idx >= 0 and idx > self.end_read_idx) or \
                idx >= len(self.txt_file_lines) - 2:
            return None
        sensor_type = SensorData.get_type(self.sensor_id)
        data = SensorData(self.sensor_id)
        if sensor_type == SensorDataType.CAMERA_FRAME.value:
            data = CameraFrameData(self.sensor_id)
        elif sensor_type == SensorDataType.LIDAR.value:
            data = LidarData(self.sensor_id)
        else:
            LOG_ERROR('SensorDataReader.read: unknown type for %d' %
                (self.sensor_id))
            return None
        if self.read_index_list:
            idx = self.read_index_list[idx]
        text_data = self.get_text_data(idx)
        if sensor_type == SensorDataType.CAMERA_FRAME.value:
            data.fidx = idx
        data.set_from_text_data(text_data) # fidx can be overwritten
        # binary file
        bin_file_idx = idx // self.num_packets_per_file
        bin_filepath = self.filepath_prefix + self.capture_bin_fmt % (
            bin_file_idx)
        bin_file = open(bin_filepath, 'rb')
        rest_read_idx = idx - (bin_file_idx * self.num_packets_per_file)
        if rest_read_idx > 0:
            bin_file.seek(
                rest_read_idx * (self.packet_size + 4), 1)
        b = bin_file.read(4)
        if len(b) != 4:
            LOG_ERROR('SensorDataReader.read: failed to read bin file')
            bin_file.close()
            return None
        packet_size = unpack('i', b)[0]
        data.data = np.fromfile(bin_file,
            dtype=np.uint8, count=packet_size)
        bin_file.close()
        data.set_meta_from_header(self.header.capture_desc)
        return data

    def close(self):
        if self.bin_file is not None: self.bin_file.close()

    def __len__(self) -> int:
        if self.end_read_idx < 0:
            LOG_WARNING('open SensorDataReader first')
            return 0
        if not self.read_index_list:
            return (self.end_read_idx - self.start_read_idx + \
                self.step_read_idx) // self.step_read_idx
        else:
            return len(self.read_index_list)
