from time import time_ns

import numpy as np

from datalink.data import PurePursuitConfig
from modules.path_tracking.pid_pure_pursuit import PurePursuit


class Controller:
    def __init__(self):
        self.timestamp = time_ns()

    def update(self) -> bool:
        raise NotImplementedError()


class DualSense(Controller):
    MAX_SPEED = 0.5  # <0,5>
    MAX_STEERING_ANGLE = 25.0
    TRIGGER_RESOLUTION = 2**8
    STICK_RESOLUTION = 2**7

    TRIGGER_FORCE2SPEED = MAX_SPEED / TRIGGER_RESOLUTION
    STICK_DEFLECT2ANGLE = MAX_STEERING_ANGLE / STICK_RESOLUTION

    def __init__(self):
        super().__init__()
        # Reduce loading time if not used
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

    def update(self) -> bool:
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
        finally:
            return self.data_ready

    def is_v_nonzero(self) -> bool:
        return not (-0.01 < self.v < 0.01)
    
    def is_sa_nonzero(self) -> bool:
        return not (-1 < self.sa < 1)

    def shutdown(self):
        if self.is_alive():
            self._dualsense.close()


class PurePursuitController(Controller):
    def __init__(self, config: PurePursuitConfig):
        super().__init__()
        self.pp = PurePursuit(config)

    def update(self, path: np.ndarray, speed: float):
        self.steering_angle = np.rad2deg(self.pp.get_control(path, speed))
        self.timestamp = time_ns()

    def set_config(self, config: PurePursuitConfig):
        self.pp.config = config