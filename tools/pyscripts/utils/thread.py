# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# utils.thread
# from cvlib's utils/thread in multipleye
# Author: Changhee Won (changhee.won@multipleye.co)
#
#
from enum import Enum
from threading import Thread, Lock, Condition

class SuspendFlag:
    class Value(Enum):
        CLEAR = 0; PAUSE = 1; STEP = -1

    def __init__(self, flag: int = Value.CLEAR.value):
        self.flag_ = flag
        self.mutex_ = Lock()
        self.cond_ = Condition(self.mutex_)

    def value(self) -> int:
        self.mutex_.acquire()
        val = self.flag_
        self.mutex_.release()
        return val

    def clear(self):
        self.mutex_.acquire()
        self.flag_ = 0
        self.cond_.notify_all()
        self.mutex_.release()

    def step(self):
        self.mutex_.acquire()
        self.flag_ = -1
        self.cond_.notify_all()
        self.mutex_.release()

    def set(self):
        self.mutex_.acquire()
        self.flag_ = 1
        self.mutex_.release()

    def toggle(self):
        self.mutex_.acquire()
        if self.flag_ > 0: self.cond_.notify_all()
        self.flag_ = 0 if self.flag_ > 0 else 1
        self.mutex_.release()

    def check(self):
        self.mutex_.acquire()
        while self.flag_ > 0: self.cond_.wait()
        if self.flag_ == -1: self.flag_ = 1
        self.mutex_.release()
