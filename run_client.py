#!/usr/bin/env python3

import threading
import queue
import numpy as np
import socket
import struct
import time
from ToySim.settings import SimulationCameraSettings as Cam, SimulationDataSizesBytes as Sizes, RenderLoopSettings, VehicleCamera 
from ToySim.processor import EncoderData, IMUData, Position, Rotation, SimulationDataFrame, VehicleDataFrame, Pose
import math
import argparse


class SimulationData:
    def __init__(self):
        self._counter:int = 0
        self._time:float = 0.0

    def __call__(self, dt:float):
        df = SimulationDataFrame()
        
        df.camera_frame_rgb = np.zeros((Cam.HEIGHT, Cam.WIDTH, Sizes.Camera.RGB_PIXEL), dtype=np.uint8)        
        df.camera_frame_rgb[self._counter : self._counter + 10, :, 1] = 255
        df.camera_frame_depth = np.zeros((Cam.HEIGHT, Cam.WIDTH), dtype=np.float16)
        df.camera_frame_depth[self._counter : self._counter + 10] = 30
        df.render_enqueued_unix_timestamp = time.time_ns()
        df.render_finished_unix_timestamp = time.time_ns()
        df.game_frame_number = self._counter
        df.render_frame_number = self._counter
        df.speed = math.sin(self._time)
        df.steering_angle = 40 * math.sin(self._time)
        df.pose = Pose(Position(.0, .0, .0), Rotation(.0, .0, .0))
        df.delta_time = dt

        self._counter = (self._counter + 5) % Cam.HEIGHT
        self._time += RenderLoopSettings.RENDER_DTIME

        return df.tobytes()

class VehicleData:
    def __init__(self):
        self._counter:int = 0
        self._time:float = 0.0

    def __call__(self, dt:float):
        _sin = math.sin(self._time)
        _cos = math.cos(self._time)
        
        df = VehicleDataFrame()
        df.camera_frame_rgb = np.zeros((VehicleCamera.HEIGHT, VehicleCamera.WIDTH, Sizes.Camera.RGB_PIXEL), dtype=np.uint8)        
        df.camera_frame_rgb[self._counter : self._counter + 10, :, 1] = 255
        df.camera_frame_depth = np.zeros((VehicleCamera.HEIGHT, VehicleCamera.WIDTH), dtype=np.float16)
        df.camera_frame_depth[self._counter : self._counter + 10] = 30
        df.motors_power = _sin
        df.speed = _sin
        df.steering_angle = 40 * _sin
        df.imu_data = IMUData(_sin, _cos, _sin, _cos, _sin, _cos)
        df.encoder_data = EncoderData(_sin, _cos)
        df.pose = Pose(Position(_cos*1000, _sin*1000, _cos*1000), Rotation(_sin, _sin, _sin))
        df.delta_time = dt

        self._counter = (self._counter + 5) % VehicleCamera.HEIGHT
        self._time += RenderLoopSettings.RENDER_DTIME

        return df.tobytes()


def network_thread(send_queue:queue.Queue, stop_event:threading.Event):
    addr = ('localhost', 8888)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(addr)
        
        def send_data():
            while not stop_event.is_set():
                data = send_queue.get()
                sock.sendall(data)

        def recv_data():
            while not stop_event.is_set():
                data = sock.recv(8)
                print(struct.unpack('ff', data))
        
        send_thread = threading.Thread(target=send_data)
        recv_thread = threading.Thread(target=recv_data)
        send_thread.start()
        recv_thread.start()

        send_thread.join()
        recv_thread.join()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--client', 
        choices=['sim', 'veh'], 
        required=True, 
        help="Client type defines structure of data exchanged with the server."
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    dummy_data = SimulationData() if args.client == 'sim' else VehicleData()
    send_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    network_t = threading.Thread(target=network_thread, args=[send_queue, stop_event], daemon=True)
    network_t.start()

    try:
        while not stop_event.is_set():
            send_queue.put(dummy_data(RenderLoopSettings.RENDER_DTIME))
    except KeyboardInterrupt:
        print("[Client] Interrupt received, shutting down...")
    finally:
        stop_event.set()
