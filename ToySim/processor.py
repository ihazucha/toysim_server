import numpy as np
import struct

from queue import Queue, Full
from threading import Event, Thread

from .settings import SimulationDataSizesBytes, SimulationCameraSettings


class Pose:
    def __init__(self, position, rotation):
        self.position = position
        self.rotation = rotation

class Position:
    def __init__(self, x:float, y:float, z:float):
        self.x = x
        self.y = y
        self.z = z

class Rotation:
    def __init__(self, roll:float, pitch:float, yaw:float):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

class SimulationDataFrame:
    STRUCT_FORMAT = [
        "Q",  # render_enqueued_unix_timestamp
        "Q",  # render_finished_unix_timestamp
        "Q",  # game_frame_number
        "Q",  # render_frame_number
        "f",  # speed
        "f",  # steering_angle
        "ddd",  # pose.position
        "ddd",  # pose.rotation
        "f", # delta_time
    ]
    STRUCT_FORMAT_STRING = "".join(STRUCT_FORMAT)
    
    def __init__(self, data_bytes: bytes):
        self.data_bytes:bytes = data_bytes
        self.camera_frame_rgb:np.ndarray = None
        self.camera_frame_depth:np.ndarray = None
        self.render_enqueued_unix_timestamp:int = None
        self.render_finished_unix_timestamp:int = None
        self.game_frame_number:int = None
        self.render_frame_number:int = None
        self.speed:float = None
        self.steering_angle:float = None
        self.pose:Pose = None
        self.delta_time:float = None
        self._decode()

    def _decode(self):
        rgb_bytes = self.data_bytes[:SimulationDataSizesBytes.Camera.RGB_FRAME]
        rgb_shape = (SimulationCameraSettings.HEIGHT, SimulationCameraSettings.WIDTH, SimulationDataSizesBytes.Camera.RGB_PIXEL)
        self.camera_frame_rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(rgb_shape)
        depth_bytes = self.data_bytes[SimulationDataSizesBytes.Camera.RGB_FRAME:SimulationDataSizesBytes.Camera.TOTAL_FRAME]
        self.camera_frame_depth = np.frombuffer(depth_bytes, dtype=np.float16)
        (
            self.render_enqueued_unix_timestamp,
            self.render_finished_unix_timestamp,
            self.game_frame_number,
            self.render_frame_number,
            self.speed,
            self.steering_angle,
            x, y, z,
            roll, pitch, yaw,
            self.delta_time
        ) = struct.unpack(self.__class__.STRUCT_FORMAT_STRING, self.data_bytes[SimulationDataSizesBytes.Camera.TOTAL_FRAME:])
        self.pose = Pose(Position(x, y, z), Rotation(roll, pitch, yaw))
        return self

class ControlDataFrame:
    def __init__(self, speed_setpoint, steering_angle_setpoint):
        self.speed_setpoint = speed_setpoint
        self.steering_angle_setpoint = steering_angle_setpoint

    def get_encoded(self):
        return struct.pack('ff', self.speed_setpoint, self.steering_angle_setpoint)

class Processor:
    def __init__(
        self,
        data_queue: Queue,
        control_queue: Queue,
        render_queue: Queue,
        connected_event: Event,
    ):
        self._data_queue = data_queue
        self._control_queue = control_queue
        self._render_queue = render_queue
        self._connected_event = connected_event
        self._thread = Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        self._connected_event.wait()
        while True:
            # Data queue
            # TODO: handle queue blocking (timeout/exception)
            data_bytes = self._data_queue.get()
            simulation_data_frame = SimulationDataFrame(data_bytes=data_bytes)

            # Control queue
            speed_setpoint = 0
            steering_angle_setpoint = 0
            control_data_frame = ControlDataFrame(speed_setpoint, steering_angle_setpoint)
            try:
                self._control_queue.put(control_data_frame.get_encoded(), block=False)
            except Full:
                pass
            
            # Render queue
            self._render_queue.put((simulation_data_frame, control_data_frame))
