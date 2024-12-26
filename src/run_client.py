#!/usr/bin/env python3

import os
import time
import math
import numpy as np

from multiprocessing import Process

from src.utils.ipc import SPMCQueue
from src.utils.data import RawImageData, SensorData, IMUData, EncoderData, Pose, Position, Rotation, RemoteControlData
from src.modules.network import NetworkClient, get_local_ip


class ImageGenerator(Process):
    def __init__(self, q_image: SPMCQueue):
        super().__init__()
        self._q_image = q_image

    def run(self):
        q = self._q_image.get_producer()
        i = 0
        h, w = 480, 640
        while True:
            img = np.zeros((h, w, 3), dtype=np.uint8)        
            img[i : i + 10, :, 1] = 255
            i = (i + 5) % h
            int_timestamp = int(time.time() * 1e6)
            img_data = RawImageData(timestamp=int_timestamp, image_array=img)
            q.put(img_data)
            time.sleep(0.016)

class SensorGenerator(Process):
    def __init__(self, q_sensor: SPMCQueue):
        super().__init__()
        self._q_sensor = q_sensor

    def run(self):
        q = self._q_sensor.get_producer()
        t = 0.0
        while True:
            ts = time.time()
            sint = math.sin(t)
            cost = math.sin(t)
            imu_data = IMUData(ts, sint, sint, sint, cost, cost, cost)
            lencoder_data = EncoderData(ts, int(t) % 4096, 50, sint) 
            rencoder_data = EncoderData(ts, int(t) % 4096, 60, cost) 
            pose_data = Pose(Position(sint, sint, sint), Rotation(cost, cost, cost))
            sensor_data = SensorData(imu_data, lencoder_data, rencoder_data, pose_data)
            t += ts
            q.put(sensor_data)
            time.sleep(0.008)

class RemoteReceiver(Process):
    def __init__(self, q_control: SPMCQueue):
        super().__init__()
        self._q_control = q_control

    def run(self):
        q = self._q_control.get_consumer()
        while True:
            remote_data = q.get()
            print(RemoteControlData.from_bytes(remote_data))
            time.sleep(0.008)


if __name__ == '__main__':
    q_image = SPMCQueue(11001)
    q_sensor = SPMCQueue(11002)
    q_remote = SPMCQueue(11003)
    
    p_image = ImageGenerator(q_image)
    p_sensor = SensorGenerator(q_sensor)
    p_remote = RemoteReceiver(q_remote)
    p_network = NetworkClient(q_image=q_image, q_sensor=q_sensor, q_remote=q_remote, server_ip=get_local_ip())

    ps = [p_image, p_sensor, p_remote, p_network]

    try:
        [p.start() for p in ps]  # type: ignore
        [p.join() for p in ps]  # type: ignore
    except (KeyboardInterrupt, SystemExit):
        print("[Client] Interrupt received, shutting down...")
        [p.terminate() for p in ps]  # type: ignore
    finally:
        [p.join() for p in ps]  # type: ignore
        os._exit(0)
