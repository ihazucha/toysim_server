import numpy as np

from struct import unpack
from queue import Queue, Full
from threading import Event, Thread

from .settings import (
    CAMERA_X,
    CAMERA_Y,
    CAMERA_PIXEL_COMPONENTS,
)


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
        # TODO: handle queue blocking (timeout/exception)
        while True:
            self._connected_event.wait()
            # Data queue
            data_bytes = self._data_queue.get()
            speed, steering_angle = unpack("ff", data_bytes[:8])
            image_data_bytes = data_bytes[8:]
            image_data_np = np.frombuffer(image_data_bytes, dtype=np.uint8)
            image_rgba = image_data_np.reshape(
                (CAMERA_Y, CAMERA_X, CAMERA_PIXEL_COMPONENTS)
            )

            # Control queue
            new_speed, new_steering_angle = speed + 0.001, steering_angle + 0.01
            try:
                self._control_queue.put((new_speed, new_steering_angle), block=False)
            except Full:
                pass
            # Render queue
            self._render_queue.put(
                (speed, new_speed, steering_angle, new_steering_angle, image_rgba)
            )
