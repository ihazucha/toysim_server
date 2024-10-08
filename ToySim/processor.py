import numpy as np
import struct

from queue import Queue, Full
from threading import Event, Thread

from .settings import SimulationDataSizesBytes, SimulationCameraSettings, VehicleCamera, VehicleDataSizes


class Position:
    STRUCT_FORMAT_STRING = '3d'
    
    def __init__(self, x:float, y:float, z:float):
        self.x = x
        self.y = y
        self.z = z

    def tobytes(self):
        return struct.pack(self.__class__.STRUCT_FORMAT_STRING, self.x, self.y, self.z)

class Rotation:
    STRUCT_FORMAT_STRING = '3d'

    def __init__(self, roll:float, pitch:float, yaw:float):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def tobytes(self):
        return struct.pack(self.__class__.STRUCT_FORMAT_STRING, self.roll, self.pitch, self.yaw)

class Pose:
    STRUCT_FORMAT_STRING = Position.STRUCT_FORMAT_STRING + Rotation.STRUCT_FORMAT_STRING
    
    def __init__(self, position:Position, rotation:Rotation):
        self.position = position
        self.rotation = rotation
    
    def tobytes(self):
        return self.position.tobytes() + self.rotation.tobytes()

class IMUData:
    STRUCT_FORMAT_STRING = '6f'

    def __init__(self, ax:float, ay:float, az:float, w_roll:float, w_pitch:float, w_yaw:float):
        self.ax = ax
        self.ay = ay
        self.az = az
        self.w_roll = w_roll
        self.w_pitch = w_pitch
        self.w_yaw = w_yaw

    def tobytes(self):
        return struct.pack(self.__class__.STRUCT_FORMAT_STRING, self.ax, self.ay, self.az, self.w_roll, self.w_pitch, self.w_yaw)

class EncoderData:
    STRUCT_FORMAT_STRING = '2f'
    def __init__(self, position:float, magnitude:float):
        self.position = position
        self.magnutide = magnitude

    def tobytes(self):
        return struct.pack(self.__class__.STRUCT_FORMAT_STRING, self.position, self.magnutide)

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
    
    def __init__(self, data_bytes:bytes=b''):
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
        
    def tobytes(self):
        self.data_bytes += self.camera_frame_rgb.tobytes()
        self.data_bytes += self.camera_frame_depth.tobytes()
        self.data_bytes += struct.pack(
            '4Q2f',
            self.render_enqueued_unix_timestamp,
            self.render_finished_unix_timestamp,
            self.game_frame_number,
            self.render_frame_number,
            self.speed,
            self.steering_angle,
        )
        self.data_bytes += self.pose.tobytes()
        self.data_bytes += struct.pack('f', self.delta_time)
        return self.data_bytes

    def unpack(self):
        self.camera_frame_rgb = np.frombuffer(self.data_bytes[:SimulationDataSizesBytes.Camera.RGB_FRAME], dtype=np.uint8)
        self.camera_frame_rgb.reshape((SimulationCameraSettings.HEIGHT, SimulationCameraSettings.WIDTH, SimulationDataSizesBytes.Camera.RGB_PIXEL))
        self.camera_frame_depth = np.frombuffer(self.data_bytes[SimulationDataSizesBytes.Camera.RGB_FRAME:SimulationDataSizesBytes.Camera.TOTAL_FRAME], dtype=np.float16)
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

class VehicleDataFrame:
    STRUCT_FORMAT = [
        '4f',
        IMUData.STRUCT_FORMAT_STRING,
        EncoderData.STRUCT_FORMAT_STRING,
        Pose.STRUCT_FORMAT_STRING
    ]
    STRUCT_FORMAT_STRING = "".join(STRUCT_FORMAT)
    
    def __init__(self, data_bytes:bytes=b''):
        self.data_bytes = data_bytes
        self.camera_frame_rgb:np.ndarray = None
        self.camera_frame_depth:np.ndarray = None
        self.motors_power:float = None
        self.speed:float = None
        self.steering_angle:float = None
        self.delta_time:float = None
        self.imu_data:IMUData = None
        self.encoder_data:EncoderData = None
        self.pose:Pose = None

    def tobytes(self):
        self.data_bytes += self.camera_frame_rgb.tobytes()
        self.data_bytes += self.camera_frame_depth.tobytes()
        self.data_bytes += struct.pack(
            '4f',
            self.motors_power,
            self.speed,
            self.steering_angle,
            self.delta_time
        )
        self.data_bytes += self.imu_data.tobytes()
        self.data_bytes += self.encoder_data.tobytes()
        self.data_bytes += self.pose.tobytes()
        return self.data_bytes

    def unpack(self):
        self.camera_frame_rgb = np.frombuffer(self.data_bytes[:VehicleDataSizes.Camera.RGB_FRAME], dtype=np.uint8)
        self.camera_frame_rgb.reshape((VehicleCamera.HEIGHT, VehicleCamera.WIDTH, VehicleDataSizes.Camera.RGB_PIXEL))
        self.camera_frame_depth = np.frombuffer(self.data_bytes[VehicleDataSizes.Camera.RGB_FRAME:VehicleDataSizes.Camera.TOTAL_FRAME], dtype=np.float16)
        (
            self.motors_power,
            self.speed,
            self.steering_angle,
            self.delta_time,
            ax, ay, az, w_roll, w_pitch, w_yaw,
            position, magnitude,
            x, y, z, roll, pitch, yaw
        ) = struct.unpack(self.__class__.STRUCT_FORMAT_STRING, self.data_bytes[VehicleDataSizes.Camera.TOTAL_FRAME:])
        self.imu_data = IMUData(ax, ay, az, w_roll, w_pitch, w_yaw)
        self.encoder_data = EncoderData(position, magnitude)
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
        _counter = 0
        import math
        while True:
            # Data queue
            # TODO: handle queue blocking (timeout/exception)
            data_bytes = self._data_queue.get()
            simulation_data_frame = SimulationDataFrame(data_bytes=data_bytes)
            simulation_data_frame.unpack()

            # Control queue
            speed_setpoint = math.sin(_counter)
            steering_angle_setpoint = math.cos(_counter)
            _counter += 1
            control_data_frame = ControlDataFrame(speed_setpoint, steering_angle_setpoint)
            try:
                self._control_queue.put(control_data_frame.get_encoded(), block=False)
            except Full:
                pass
            
            # Render queue
            self._render_queue.put((simulation_data_frame, control_data_frame))
