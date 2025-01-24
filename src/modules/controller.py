from pydualsense import pydualsense
from utils.data import RemoteControlData
from modules.path_tracking.pid_pure_pursuit import PIDController, PurePursuit

import time
import numpy as np


class Controller:
    def get_inputs(self) -> tuple[float, float]:
        raise NotImplementedError()

    def is_alive(self) -> bool:
        raise NotImplementedError()

class DualSense(Controller):
    MAX_POWER = 0.2  # <0,1>
    MAX_STEERING_ANGLE = 25.0  # Both sides
    TRIGGER_BUTTON_RESOLUTION = 2**8
    STICK_RESOLUTION = 2**7
    BTN_FORCE2POWER = MAX_POWER / TRIGGER_BUTTON_RESOLUTION
    STICK_FORCE2POWER = MAX_STEERING_ANGLE / STICK_RESOLUTION
    CONTROLLER_STICK_OFFSET = MAX_STEERING_ANGLE / STICK_RESOLUTION

    def __init__(self):
        self._dualsense = pydualsense()
        self._connected = False
        try:
            self._dualsense.init()
            self._connected = True
            self._dualsense.light.setColorI(0, 255, 0)
        except:
            print("[DualSense] Unable to connect.")

    def get_inputs(self):
        # Brake - Gas = Force
        force_diff = self._dualsense.state.L2 - self._dualsense.state.R2
        set_speed = DualSense.BTN_FORCE2POWER * force_diff
        set_steering_angle = DualSense.STICK_FORCE2POWER * self._dualsense.state.LX + DualSense.CONTROLLER_STICK_OFFSET
        return RemoteControlData(time.time_ns(), set_speed, set_steering_angle)

    def is_alive(self):
        return self._connected

    def close(self):
        if self.isAlive():
            self._dualsense.close()

class PurePursuitPIDController:
    cm2m = 1e2

    def __init__(self, pure_pursuit=PurePursuit(), pid=PIDController(Kp=2, Ki=0, Kd=0)):
        self.pure_pursuit = pure_pursuit
        self.pid = pid

    def get_inputs(self, path: np.ndarray, speed_cmps: float, dt: float):
        path_m = path / self.__class__.cm2m
        speed_ms = speed_cmps / self.__class__.cm2m
        self.set_speed = self.pid.get_control(measurement=speed_ms, set_point=10.0, dt=dt)
        self.set_steering_angle = np.rad2deg(self.pure_pursuit.get_control(path_m, speed_ms))
        return (self.set_speed, self.set_steering_angle)
    
    def is_alive(self):
        return True