from pydualsense import pydualsense  # type: ignore
from utils.data import RemoteControlData  # type: ignore
from multiprocessing import Queue

import time


class Controller:
    def get_input(self):
        NotImplementedError()

    def is_connected(self):
        NotImplementedError()


class DualSense(Controller):
    MAX_POWER = 0.2  # <0,1>
    MAX_STEERING_ANGLE = 25  # Both sides
    TRIGGER_BUTTON_RESOLUTION = 2**8
    STICK_RESOLUTION = 2**7
    BTN_FORCE2POWER = MAX_POWER / TRIGGER_BUTTON_RESOLUTION
    STICK_FORCE2POWER = MAX_STEERING_ANGLE / STICK_RESOLUTION

    def __init__(self, control_queue: Queue):
        self._control_queue = control_queue
        self._dualsense = pydualsense()
        self._connected = False
        try:
            self._dualsense.init()
            self._connected = True
            self._dualsense.light.setColorI(0, 255, 0)
        except:
            print("[DualSense] Unable to connect.")

    def get_input(self):
        # Brake - Gas = Force
        force_diff = self._dualsense.state.L2 - self._dualsense.state.R2
        set_speed = DualSense.BTN_FORCE2POWER * force_diff
        set_steering_angle = DualSense.STICK_FORCE2POWER * self._dualsense.state.LX
        return RemoteControlData(time.time(), set_speed, set_steering_angle)

    def is_connected(self):
        return self._connected

    def close(self):
        if self.isAlive():
            self._dualsense.close()
