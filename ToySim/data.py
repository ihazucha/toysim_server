import struct
from typing import Type


class SerializableMeta(type):
    """Just to make sure every data class is correctly defined"""

    required_attributes: list[str] = []

    def __new__(cls, name, bases, dct):
        new_cls = super().__new__(cls, name, bases, dct)
        for attr in cls.required_attributes:
            if not hasattr(new_cls, attr):
                raise TypeError(f"Class {name} must define {attr} attribute")
        return new_cls


class Serializable(metaclass=SerializableMeta):
    required_attributes = ["SIZE"]
    SIZE: int = 0

    @classmethod
    def from_bytes(cls, data: bytes):
        raise NotImplementedError()

    def to_bytes(self) -> bytes:
        raise NotImplementedError()

    def to_list(self) -> list:
        raise NotImplementedError()


class SerializablePrimitive(Serializable):
    required_attributes = ["SIZE", "FORMAT"]
    FORMAT: str = ""

    @classmethod
    def from_bytes(cls, data: bytes):
        return cls(*struct.unpack(cls.FORMAT, data))

    def to_bytes(self):
        return struct.pack(self.__class__.FORMAT, *self.to_list())


class SerializableComplex(Serializable):
    required_attributes = ["SIZE", "COMPONENTS"]
    COMPONENTS: list[Type[Serializable]] = []

    @classmethod
    def from_bytes(cls, data: bytes):
        p = 0
        objects = []
        for c in cls.COMPONENTS:
            c_data = data[p : p + c.SIZE]
            o = c.from_bytes(c_data)
            objects.append(o)
            p += c.SIZE
        return cls(*objects)

    def to_bytes(self):
        return b''.join([c.to_bytes() for c in self.to_list()])


# -------------------------------------------------------------------------------------------------


class Position(SerializablePrimitive):
    FORMAT = "=3d"
    SIZE = struct.calcsize(FORMAT)

    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z

    def to_list(self):
        return [self.x, self.y, self.z]


class Rotation(SerializablePrimitive):
    FORMAT = "=3d"
    SIZE = struct.calcsize(FORMAT)

    def __init__(self, roll: float, pitch: float, yaw: float):
        self.roll, self.pitch, self.yaw = roll, pitch, yaw

    def to_list(self):
        return [self.roll, self.pitch, self.yaw]


class Pose(SerializableComplex):
    COMPONENTS = [Position, Rotation]
    SIZE = sum([c.SIZE for c in COMPONENTS])

    def __init__(self, position: Position, rotation: Rotation):
        self.position = position
        self.rotation = rotation

    def to_list(self):
        return [self.position, self.rotation]


class IMUData(SerializablePrimitive):
    FORMAT = "=7d"
    SIZE = struct.calcsize(FORMAT)

    def __init__(
        self,
        timestamp: float,
        ax: float,
        ay: float,
        az: float,
        wroll: float,
        wpitch: float,
        wyaw: float,
    ):
        self.timestamp = timestamp
        self.ax = ax
        self.ay = ay
        self.az = az
        self.wroll = wroll
        self.wpitch = wpitch
        self.wyaw = wyaw

    def to_list(self):
        return [
            self.timestamp,
            self.ax,
            self.ay,
            self.az,
            self.wroll,
            self.wpitch,
            self.wroll,
        ]


class EncoderData(SerializablePrimitive):
    FORMAT = "=diid"
    SIZE = struct.calcsize(FORMAT)

    def __init__(self, timestamp: float, position: int, magnitude: int, speed: float):
        self.timestamp = timestamp
        self.position = position
        self.magnitude = magnitude
        self.speed = speed

    def to_list(self):
        return [self.timestamp, self.position, self.magnitude, self.speed]


class ActuatorsData(SerializablePrimitive):
    FORMAT = "=4d"
    SIZE = struct.calcsize(FORMAT)

    def __init__(
        self,
        timestamp: float,
        motor_power: float,
        steering_angle: float,
        speed: float,
    ):
        self.timestamp = timestamp
        self.motor_power = motor_power
        self.steering_angle = steering_angle
        self.speed = speed

    def to_list(self):
        return [self.timestamp, self.motor_power, self.steering_angle, self.speed]


class SensorData(SerializableComplex):
    COMPONENTS = [IMUData, EncoderData, EncoderData, Pose]
    SIZE = sum([c.SIZE for c in COMPONENTS])

    def __init__(
        self,
        imu: IMUData,
        rleft_encoder: EncoderData,
        rright_encoder: EncoderData,
        pose: Pose,
    ):
        self.imu = imu
        self.rleft_encoder = rleft_encoder
        self.rright_encoder = rright_encoder
        self.pose = pose

    def to_list(self):
        return [self.imu, self.rleft_encoder, self.rright_encoder, self.pose]


class VehicleControlData(SerializablePrimitive):
    FORMAT = "=3d"
    SIZE = struct.calcsize(FORMAT)

    def __init__(self, timestamp: float, set_speed: float, set_steering_angle: float):
        self.timestamp = timestamp
        self.set_speed = set_speed
        self.set_steering_angle = set_steering_angle

    def to_list(self):
        return [self.timestamp, self.set_speed, self.set_steering_angle]
