import numpy as np
import cv2

from typing import Any

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from datalink.ipc import messaging
from utils.env import Environment, ENV
from datalink.data import ProcessedData, JPGImageData
from cv2 import imdecode, IMREAD_COLOR


class QSimData:
    def __init__(self, data: ProcessedData):
        self.data: ProcessedData = data
        self.processed_rgb_qimage: QImage = None
        self.processed_depth_qimage: QImage = None


# TODO: make more explicit what is the format of this image
def npimage2qimage(npimage: np.ndarray[Any, np.dtype[np.uint8]]):
    height, width, channel = npimage.shape
    bytes_per_line = channel * width
    return QImage(npimage.data, width, height, bytes_per_line, QImage.Format_RGB888)


def depth2qimage(depth: np.ndarray) -> QImage:
    depth_colormap = depth_to_colormap(depth)
    h, w = depth_colormap.shape[1], depth_colormap.shape[0]
    return QImage(depth_colormap.data, h, w, QImage.Format_BGR888)


def depth_to_colormap(depth_data: np.ndarray):
    normalized_inverted = 255 - (depth_data / 5000 * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(normalized_inverted, cv2.COLORMAP_INFERNO)
    return colormap


class ImageDataThread(QThread):
    simulation_data_ready = Signal(QSimData)
    data_ready = Signal(tuple)  # TODO: rework for vehicle - refactor to common standard

    def __init__(self):
        super().__init__()
        self._is_running = True

    def run(self):
        if ENV == Environment.VEHICLE:
            self._run_vehicle()
        elif ENV == Environment.SIM:
            self._run_sim()
        else:
            raise NotImplementedError(f"No runtime found for ENV={ENV}")

    def _run_vehicle(self):
        q = messaging.q_image.get_consumer()
        while self._is_running:
            jpg_image_data: JPGImageData = q.get()
            image_array = imdecode(np.frombuffer(jpg_image_data.jpg, np.uint8), IMREAD_COLOR)
            qimage = npimage2qimage(image_array)
            self.data_ready.emit((qimage, jpg_image_data.timestamp))

    def _run_sim(self):
        q_processing = messaging.q_processing.get_consumer()
        while self._is_running:
            data: ProcessedData = q_processing.get()

            qsim_data = QSimData(data=data)
            qsim_data.processed_rgb_qimage = npimage2qimage(data.debug_image)
            qsim_data.processed_depth_qimage = depth2qimage(data.depth)

            self.simulation_data_ready.emit(qsim_data)

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
