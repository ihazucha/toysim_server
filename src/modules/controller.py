from time import time_ns

import numpy as np

from datalink.data import PurePursuitPIDConfig
from modules.path_tracking.pid_pure_pursuit import PID, PurePursuit


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
        # TODO: adhoc import to reduce lib loading time if not used
        from pydualsense import pydualsense
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


class PurePursuitPID(Controller):
    def __init__(self, config: PurePursuitPIDConfig):
        super().__init__()
        self.pp = PurePursuit()
        self.pid = PID(Kp=1.5, Ki=0, Kd=0)
        self.set_config(config)

    def update(self, path: np.ndarray, speed: float, dt: float):
        self.speed = self.pid.update(measured=speed, desired=self.v_setpoint, dt=dt)
        self.steering_angle = np.rad2deg(self.pp.get_control(path, speed))
        self.timestamp = time_ns()

    def set_config(self, config: PurePursuitPIDConfig):
        self.pp.lookahead_factor = config.lookahead_factor
        self.pp.lookahead_l_min = config.lookahead_l_min
        self.pp.lookahead_l_max = config.lookahead_l_max
        self.v_setpoint = config.speed_setpoint

    def is_alive(self):
        return True
