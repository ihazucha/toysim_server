#!/usr/bin/env python3

import threading
import queue
import numpy as np
import socket
import os
import struct
import time
from datetime import datetime
from ToySim.settings import SimulationCameraSettings as Cam, SimulationDataSizesBytes as Sizes, RenderLoopSettings, NetworkSettings, VehicleCamera, ClientTypes
from ToySim.processor import EncoderData, IMUData, Position, Rotation, SimulationDataFrame, VehicleDataFrame, Pose
import math
import signal

stop_threads = False

def signal_handler(sig, frame):
    global stop_threads
    stop_threads = True

signal.signal(signal.SIGINT, signal_handler)

DATA_QUEUE = queue.Queue(maxsize=1)

class SimulationDummyData:
    def __init__(self):
        self._counter = 0
        self._time = 0

    def __call__(self):
        generator_start = datetime.now().timestamp()
        
        data_frame = SimulationDataFrame()
        
        data_frame.camera_frame_rgb = np.zeros((Cam.HEIGHT, Cam.WIDTH, Sizes.Camera.RGB_PIXEL), dtype=np.uint8)        
        data_frame.camera_frame_rgb[self._counter : self._counter + 10, :, 1] = 255
        data_frame.camera_frame_depth = np.zeros((Cam.HEIGHT, Cam.WIDTH), dtype=np.float16)
        data_frame.camera_frame_depth[self._counter : self._counter + 10] = 30

        data_frame.render_enqueued_unix_timestamp = time.time_ns()
        data_frame.render_finished_unix_timestamp = time.time_ns()
        data_frame.game_frame_number = self._counter
        data_frame.render_frame_number = self._counter
        data_frame.speed = math.sin(self._time)
        data_frame.steering_angle = 40 * math.sin(self._time)
        data_frame.pose = Pose(Position(.0, .0, .0), Rotation(.0, .0, .0))
        time.sleep(1/60)
        data_frame.delta_time = generator_start - datetime.now().timestamp()

        self._counter = (self._counter + 5) % Cam.HEIGHT
        self._time += RenderLoopSettings.RENDER_DTIME

        return data_frame.tobytes()

class VehicleDummyData:
    """ CameraData 
        GyroDataRaw ([ax, ay, az], [v_roll, a_pitch, a_yaw])
    """
    def __init__(self):
        self._counter = 0
        self._time = 0

    def __call__(self):
        generator_start = datetime.now().timestamp()
        _sin = math.sin(self._time)
        _cos = math.cos(self._time)
        data_frame = VehicleDataFrame()
        
        data_frame.camera_frame_rgb = np.zeros((VehicleCamera.HEIGHT, VehicleCamera.WIDTH, Sizes.Camera.RGB_PIXEL), dtype=np.uint8)        
        data_frame.camera_frame_rgb[self._counter : self._counter + 10, :, 1] = 255
        data_frame.camera_frame_depth = np.zeros((VehicleCamera.HEIGHT, VehicleCamera.WIDTH), dtype=np.float16)
        data_frame.camera_frame_depth[self._counter : self._counter + 10] = 30

        data_frame.motors_power = _sin
        data_frame.speed = _sin
        data_frame.steering_angle = 40 * _sin
        data_frame.imu_data = IMUData(_sin, _cos, _sin, _cos, _sin, _cos)
        data_frame.encoder_data = EncoderData(_sin, _cos)
        data_frame.pose = Pose(Position(_cos*1000, _sin*1000, _cos*1000), Rotation(_sin, _sin, _sin))
        time.sleep(1/60)
        data_frame.delta_time = generator_start - datetime.now().timestamp()

        self._counter = (self._counter + 5) % VehicleCamera.HEIGHT
        self._time += RenderLoopSettings.RENDER_DTIME

        return data_frame.tobytes()


def tcp_data_sender(client):
    addr = ("192.168.0.104", 3333) if client == ClientTypes.SIMULATION else NetworkSettings.Vehicle.SERVER_ADDR
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(addr)
        
        def send_data():
            while not stop_threads:
                data = DATA_QUEUE.get()
                s.sendall(data)

        def recv_data():
            while not stop_threads:
                data = s.recv(8,)
                print(struct.unpack('ff', data))
        
        # Create and start the sending thread
        send_thread = threading.Thread(target=send_data)
        send_thread.start()

        # Create and start the receiving thread
        recv_thread = threading.Thread(target=recv_data)
        recv_thread.start()

        # Wait for both threads to finish
        send_thread.join()
        recv_thread.join()


def simulation_thread():
    data = SimulationDummyData()
    while not stop_threads:
        DATA_QUEUE.put(data())

def vehicle_thread():
    data = VehicleDummyData()
    while not stop_threads:
        DATA_QUEUE.put(data())

def data_thread(client):
    try:
        tcp_data_sender(client)
    except Exception as e:
        print(f"[Data] Exception occured: {e}")
        os._exit(1)

import argparse
parser = argparse.ArgumentParser(description="Run the ToySim server.")
parser.add_argument(
    '-c', '--client', 
    choices=['sim', 'veh'], 
    required=True, 
    help="Specify the client type: 'sim' or 'veh'."
)
args = parser.parse_args()
client_map = {'sim': ClientTypes.SIMULATION, 'veh': ClientTypes.VEHICLE}
client = client_map[args.client]

data_t = threading.Thread(target=data_thread, args=[client])
data_t.daemon = True


try:
    data_t.start()
    if client == ClientTypes.SIMULATION:
        simulation_thread()
    else:
        vehicle_thread()
except KeyboardInterrupt:
    print("[Client] Interrupt request - exiting program")
finally:
    pass
