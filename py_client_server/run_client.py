#!/usr/bin/env python3

import threading
import queue
import numpy as np
import socket
import os
import time
import struct
from RoboSim.settings import CAMERA_PIXEL_COMPONENTS, CAMERA_X, CAMERA_Y, SERVER_ADDR, RENDER_DTIME

DATA_QUEUE = queue.Queue(maxsize=1)


class DummyData:
    def __init__(self):
        self._counter = 0
        self._time_step = 0

    def __call__(self):
        data = np.zeros((CAMERA_Y, CAMERA_X, CAMERA_PIXEL_COMPONENTS), dtype=np.uint8)
        data[self._counter : self._counter + 10, :, 1] = 255
        self._counter = (self._counter + 5) % CAMERA_Y

        speed = np.sin(self._time_step)
        steering_angle = 40 * np.sin(self._time_step)

        self._time_step += RENDER_DTIME

        data_bytes = b""
        data_bytes += struct.pack("ff", speed, steering_angle)
        data_bytes += data.tobytes()

        return data_bytes


def tcp_data_sender():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(SERVER_ADDR)
        while True:
            data = DATA_QUEUE.get()
            s.sendall(data)


def game_thread():
    data = DummyData()
    while True:
        DATA_QUEUE.put(data())


def data_thread():
    try:
        tcp_data_sender()
    except Exception as e:
        print(f"[Data] Exception occured: {e}")
        os._exit(1)


data_t = threading.Thread(target=data_thread)
data_t.daemon = True

try:
    data_t.start()
    while True:
        game_thread()
except KeyboardInterrupt:
    print("[Client] Interrupt request - exiting program")
finally:
    pass
