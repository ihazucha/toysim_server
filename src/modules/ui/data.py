import numpy as np
import cv2 

from typing import Any

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from utils.ipc import messaging
from utils.env import Environment, ENV
from utils.data import ControllerData, JPGImageData, SimData
from utils.image import jpg_decode


class ImageDataThread(QThread):
    data_ready = Signal(tuple)

    def __init__(self):
        super().__init__()
        self._is_running = True

    def run(self):
        if ENV == Environment.VEHICLE:
            self._run_vehicle()
        elif ENV == Environment.SIM:
            self._run_sim()
        else:
            raise NotImplementedError()

    def _run_vehicle(self):
        q = messaging.q_image.get_consumer()
        while self._is_running:
            jpg_image_data: JPGImageData = q.get()
            image_array = jpg_decode(jpg_image_data.jpg)
            qimage = __class__.ndarray2qimage(image_array)
            self.data_ready.emit((qimage, jpg_image_data.timestamp))

    def _run_sim(self):
        q = messaging.q_simulation.get_consumer()
        q_processing = messaging.q_processing.get_consumer()
        while self._is_running:
            sim_data_bytes = q.get()
            sim_data: SimData = SimData.from_bytes(sim_data_bytes)

            # rgb_image = sim_data.camera_data.rgb_image
            # TODO: figure out another to add path and intersection
            processed_data: ControllerData = q_processing.get()
            rgb_image = processed_data.image
            # --------------------------------------------------------------------

            qimage = __class__.ndarray2qimage(rgb_image)
            self.data_ready.emit((qimage, sim_data.camera_data.render_enqueued_unix_timestamp))

    @staticmethod
    def ndarray2qimage(arr: np.ndarray[Any, np.dtype[np.uint8]]):
        height, width, channel = arr.shape
        bytes_per_line = channel * width
        return QImage(arr.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def stop(self):
        self._is_running = False


class SensorDataThread(QThread):
    data_ready = Signal(tuple)

    def __init__(self):
        super().__init__()
        self._is_running = True

    def run(self):
        q_sensor = messaging.q_sensor.get_consumer()
        while self._is_running:
            data = q_sensor.get()
            self.data_ready.emit(data)

    def stop(self):
        self._is_running = False
