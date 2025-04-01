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
    MAX_SPEED = 0.5  # <0,5>
    MAX_STEERING_ANGLE = 25.0
    TRIGGER_RESOLUTION = 2**8
    STICK_RESOLUTION = 2**7

    TRIGGER_FORCE2SPEED = MAX_SPEED / TRIGGER_RESOLUTION
    STICK_DEFLECT2ANGLE = MAX_STEERING_ANGLE / STICK_RESOLUTION

    def __init__(self):
        super().__init__()
        # TODO: adhoc import to reduce lib loading time if not used
        from pydualsense import pydualsense
        self._dualsense = pydualsense()
        self._connected = False
        self.v = 0.0
        self.sa = 0.0
        self.timestamp = time_ns()
        self.data_ready = False

    def __str__(self):
        return str({k: v for k, v in self.__dict__.items() if not k.startswith('_')})

    def connect(self):
        try:
            print("connecting")
            self._dualsense.init()
            print("connected")
            self._connected = True
            self._dualsense.light.setColorI(0, 255, 0)
        except:
            print("[DualSense] Unable to connect.")

    def update(self):
        self.data_ready = False
        try:
            forward_speed = self._dualsense.state.L2
            backward_speed = self._dualsense.state.R2
            self.v = DualSense.TRIGGER_FORCE2SPEED * (forward_speed - backward_speed)
            self.sa = DualSense.STICK_DEFLECT2ANGLE * self._dualsense.state.LX
            self.timestamp = time_ns()
            self.data_ready = True
        except:
            pass

    def is_alive(self):
        # TODO: check for status dynamically
        return self._connected

    def shutdown(self):
        if self.is_alive():
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

class PurePursuitController(Controller):
    def __init__(self, config: PurePursuitPIDConfig):
        super().__init__()
        self.pp = PurePursuit(wheel_base=0.185, waypoint_shift=0.245)
        self.set_config(config)

    def update(self, path: np.ndarray, speed: float):
        self.steering_angle = np.rad2deg(self.pp.get_control(path, speed))
        self.timestamp = time_ns()

    # TODO: change to PurePursuitConfig - update UI and split the configs into 2
    def set_config(self, config: PurePursuitPIDConfig):
        self.pp.lookahead_factor = config.lookahead_factor
        self.pp.lookahead_l_min = config.lookahead_l_min
        self.pp.lookahead_l_max = config.lookahead_l_max

    def is_alive(self):
        return True
