from pydualsense import pydualsense
from datalink.data import UIConfigData
from modules.path_tracking.pid_pure_pursuit import PID, PurePursuit

from time import time_ns
import numpy as np


class Controller:
    def __init__(self):
        self.v = 0.0
        self.sa = 0.0
        self.timestamp = time_ns()

    def update(self) -> tuple[float, float]:
        raise NotImplementedError()

    def is_alive(self) -> bool:
        raise NotImplementedError()


# TODO: v is implemented as power here, when it should be in m/s
#       the internal power management should be left on the actual
#       car/simulation to make this a common interface
class DualSense(Controller):
    MAX_POWER = 0.2  # <0,1>
    MAX_STEERING_ANGLE = 25.0
    TRIGGER_RESOLUTION = 2**8
    STICK_RESOLUTION = 2**7

    TRIGGER_FORCE2POWER = MAX_POWER / TRIGGER_RESOLUTION
    STICK_DEFLECT2ANGLE = MAX_STEERING_ANGLE / STICK_RESOLUTION

    def __init__(self):
        super().__init__()
        self._dualsense = pydualsense()
        self._connected = False
        try:
            self._dualsense.init()
            self._connected = True
            self._dualsense.light.setColorI(0, 255, 0)
        except:
            print("[DualSense] Unable to connect.")

    def update(self):
        forward_power = self._dualsense.state.L2
        backward_power = self._dualsense.state.R2
        self.v = DualSense.TRIGGER_FORCE2POWER * (forward_power - backward_power)
        self.sa = DualSense.STICK_DEFLECT2ANGLE * self._dualsense.state.LX
        self.timestamp = time_ns()

    def is_alive(self):
        # TODO: check for status dynamically
        return self._connected

    def shutdown(self):
        if self.isAlive():
            self._dualsense.close()


class PurePursuitPIDController(Controller):
    def __init__(self, pure_pursuit=PurePursuit(), pid=PID(Kp=1.5, Ki=0, Kd=0)):
        super().__init__()
        self.pure_pursuit = pure_pursuit
        self.pid = pid
        # TODO: instead of UIConfigData, create an interface and send this during initialisation
        self.v_desired = UIConfigData.SET_SPEED

    def update(self, path: np.ndarray, v: float, dt: float):
        self.v = self.pid.get_control(measured=v, desired=self.v_desired, dt=dt)
        self.sa = np.rad2deg(self.pure_pursuit.get_control(path, v))
        self.timestamp = time_ns()

    def update_config(self, config: UIConfigData):
        self.pure_pursuit.K_dd = config.kdd
        self.pure_pursuit.la_clip_low = config.clip_low
        self.pure_pursuit.la_clip_high = config.clip_high
        self.speed_setpoint = config.set_speed

    def is_alive(self):
        return True
