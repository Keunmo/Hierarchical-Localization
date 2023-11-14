# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# utils.viewer.py
#
# Author: Changhee Won (changhee.won@multipleye.co)
#
#
import math
from queue import Queue, Empty
from threading import Thread, Lock, Condition
from collections import deque

from utils.common import *
from utils.open3d import *
from utils.thread import SuspendFlag

DEFINE_double('viewer_fov', 90.0, 'viewer fov')
DEFINE_integer('viewer_height', 720, 'height of viewer')
DEFINE_bool('viewer_hide_image', False, 'flag to hide image window')
DEFINE_bool('viewer_hide_geometry', False, 'flag to hide geometry window')
DEFINE_integer('viewer_msg_list_max_size', 30,
               'maximum size of viewer message list')
DEFINE_double('viewer_view_height', 20.0, 'viewer camera height')

class O3DViewerMessage:
    def __init__(self):
        self.image: np.ndarray = None
        self.geom_list: List['o3d.geometry.Geometry'] = []
        self.label_list: List[Tuple[np.ndarray, str]] = []

class O3DViewer:
    def __init__(self, title: str='viewer'):
        self.app_ = o3d_vis.gui.Application.instance
        self.app_.initialize()
        self.msgq_ = Queue()
        self.image_win_: Optional['o3d.visualization.gui.Window'] = None
        self.geom_win_: Optional['o3d.visualization.gui.Window'] = None
        self.is_running_ = False
        self.flag_ = SuspendFlag()
        self.msg_list_ = deque()

    def run(self) -> bool:
        msg: O3DViewerMessage = self.msgq_.get() 
        if msg.image is None: FLAGS.viewer_hide_image = True
        if not msg.geom_list and not msg.label_list: 
            FLAGS.viewer_hide_geometry = True
        win_h = FLAGS.viewer_height
        if not FLAGS.viewer_hide_image:
            img_w, img_h = msg.image.shape[1], msg.image.shape[0]
            img_aspect_ratio = float(img_w) / img_h
            img_win_w = math.ceil(win_h * img_aspect_ratio)
            self.image_win_ = self.app_.create_window(
                'Image', img_win_w, win_h) 
            self.image_widget_ = o3d_vis.gui.ImageWidget()
            self.image_win_.add_child(self.image_widget_)
            self.__update_image(msg.image)
            self.image_win_.set_on_close(self.__close_win) 

        if not FLAGS.viewer_hide_geometry:
            geom_win_w = round((4 / 3.0) * win_h)
            self.geom_win_ = o3d_vis.O3DVisualizer(
                'Geometry', geom_win_w, win_h) 
            for i, g in enumerate(msg.geom_list):
                self.geom_win_.add_geometry('geom#%d' % (i), g)
            self.geom_win_.show_ground = True 
            self.geom_win_.show_skybox(False) 
            self.geom_win_.enable_raw_mode(True)
            self.geom_win_.add_action(
                'Pause/Resume', self.__pause_or_resume_action)
            self.geom_win_.add_action('Step', self.__step_msg_action)
            self.geom_win_.add_action('Prev', self.__prev_msg_action)
            self.geom_win_.add_action('Next', self.__next_msg_action)
            self.geom_win_.add_action('Reset Camera', self.__reset_camera)
            self.app_.add_window(self.geom_win_)
            # self.geom_win_.reset_camera_to_default()
            self.__reset_camera(self.geom_win_)
            self.geom_win_.set_on_close(self.__close_win) 

        if not FLAGS.viewer_hide_image and not FLAGS.viewer_hide_geometry:
            ix, iy = self.image_win_.os_frame.x, self.image_win_.os_frame.y
            iw = self.image_win_.os_frame.width
            ih = self.image_win_.os_frame.height
            gw = self.geom_win_.os_frame.width
            gh = self.geom_win_.os_frame.height
            cx = ix + (iw // 2)
            center = (iw + gw) // 2
            ix = cx - center
            self.image_win_.os_frame = o3d_vis.gui.Rect(ix, iy, iw, ih)
            self.geom_win_.os_frame = o3d_vis.gui.Rect(ix + iw, iy, gw, gh)

        self.current_msg_ = msg
        self.lastest_msg_ = msg
        self.msg_list_.appendleft(msg)
        self.is_running_ = True
        self.render_thread_ = Thread(target=self.__render_thread) 
        self.render_thread_.start()

        self.app_.run() 

        self.is_running_ = False
        if self.render_thread_.is_alive(): self.render_thread_.join()

        return True

    def push(self, msg: O3DViewerMessage):
        self.msgq_.put(msg)

    def __reset_camera(self, win):
        if FLAGS.viewer_hide_geometry: return
        self.geom_win_.setup_camera(
            FLAGS.viewer_fov, [0, 0, 0],
            [0, -FLAGS.viewer_view_height, 0], [0, 0, 1] )

    def __key_callback(self, event, key):
        LOG_INFO('key%d: ', key)

    def __pause_or_resume_action(self, win): self.flag_.toggle()

    def __next_msg_action(self, win):
        idx = max(0, self.msg_list_.index(self.current_msg_) - 1)
        self.current_msg_ = self.msg_list_[idx]

    def __prev_msg_action(self, win):
        idx = min(
            len(self.msg_list_) - 1,
            self.msg_list_.index(self.current_msg_) + 1)
        self.current_msg_ = self.msg_list_[idx]

    def __step_msg_action(self, win):
        self.flag.step()

    def __update_image(self, image: np.ndarray) -> None:
        if FLAGS.viewer_hide_image: return
        self.image_widget_.update_image(o3d.geometry.Image(image))

    def __update_geometry_list(self, geom_list: List) -> None:
        if FLAGS.viewer_hide_geometry: return
        for i, g in enumerate(geom_list):
            gid = 'geom#%d' % (i)
            if i < len(self.lastest_msg_.geom_list):
                self.geom_win_.remove_geometry(gid)
                self.geom_win_.add_geometry(gid, g)
            else:
                self.geom_win_.add_geometry(gid, g)

    def __update_label_list(self, label_list: List):
        if FLAGS.viewer_hide_geometry: return
        self.geom_win_.clear_3d_labels()
        for l in label_list:
            self.geom_win_.add_3d_label(l[0], l[1])

    def __close_win(self):
        self.close()
        return True

    def __render_thread(self):
        while self.is_running_:
            try:
                msg: O3DViewerMessage = self.msgq_.get(timeout = 0.01)
            except Empty:
                msg = None
            if msg is not None:
                self.msg_list_.appendleft(msg)
                if len(self.msg_list_) > FLAGS.viewer_msg_list_max_size:
                    self.msg_list_.pop()
                self.current_msg_ = self.msg_list_[0]
            msg = self.current_msg_
            def update_image_win():
                if not FLAGS.viewer_hide_image:
                    self.__update_image(msg.image)
            def update_geometry_win():
                if not FLAGS.viewer_hide_geometry:
                    self.__update_geometry_list(msg.geom_list)
                    self.__update_label_list(msg.label_list)
            self.app_.post_to_main_thread(
                self.image_win_, update_image_win)
            self.app_.post_to_main_thread(
                self.geom_win_, update_geometry_win)
            self.lastest_msg_ = msg
            time.sleep(0.03)
        self.flag_.clear()

    def close(self):
        self.is_running_ = False
        self.flag_.clear()
        if self.render_thread_ is not None and \
                self.render_thread_.is_alive():
            self.render_thread_.join()
        self.app_.quit()
        with self.msgq_.mutex:
            self.msgq_.queue.clear()
            self.msgq_.not_full.notify_all()
            self.msgq_.not_empty.notify_all()

    @property
    def msgq(self) -> Queue: return self.msgq_
    @property
    def flag(self) -> SuspendFlag: return self.flag_
    @property
    def is_running(self) -> bool: return self.is_running_





