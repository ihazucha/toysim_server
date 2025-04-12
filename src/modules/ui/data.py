import numpy as np
import cv2

from typing import Any
from time import time_ns, sleep

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from modules.messaging import messaging
from datalink.data import ProcessedSimData, ProcessedRealData, JPGImageData, RealData
from cv2 import imdecode, IMREAD_COLOR


class QSimData:
    def __init__(self, raw: ProcessedSimData):
        self.raw = raw
        self.rgb_qimage: QImage = None
        self.depth_qimage: QImage = None
        self.processor_period_ns: float = 0
        self.processor_dt_ns: float = 0

class QRealData:
    def __init__(self, raw: ProcessedRealData):
        self.raw = raw
        self.rgb_qimage: QImage = None
        self.rgb_updated_qimage: QImage = None
        self.processor_period_ns: float = 0
        self.processor_dt_ns: float = 0

def npimage2qimage(npimage: np.ndarray[Any, np.dtype[np.uint8]]):
    h, w, channels = npimage.shape
    return QImage(npimage.data, w, h, channels * w, QImage.Format_RGB888)


def depth2qimage(depth: np.ndarray) -> QImage:
    depth_colormap = depth_to_colormap(depth)
    w, h, _ = depth_colormap.shape
    return QImage(depth_colormap.data, h, w, QImage.Format_BGR888)


def depth_to_colormap(depth_data: np.ndarray):
    normalized_inverted = 255 - (depth_data / 5000 * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(normalized_inverted, cv2.COLORMAP_INFERNO)
    return colormap


class SimDataThread(QThread):
    data_ready = Signal(QSimData)

    def __init__(self):
        super().__init__()

    def run(self):
        q_processing = messaging.q_processing.get_consumer()
        last_put_timestamp = time_ns()
        self._is_running = True
        
        while self._is_running and not self.isInterruptionRequested():
            data: ProcessedSimData = q_processing.get(100)
            if data is None:
                continue
            qsim_data = QSimData(raw=data)
            qsim_data.rgb_qimage = npimage2qimage(data.debug_image)
            qsim_data.depth_qimage = depth2qimage(data.depth)
            qsim_data.processor_period_ns = q_processing.last_put_timestamp - last_put_timestamp
            qsim_data.processor_dt_ns = q_processing.last_put_timestamp - data.begin_timestamp
            last_put_timestamp = q_processing.last_put_timestamp

            self.data_ready.emit(qsim_data)

    def stop(self):
        self._is_running = False
        self.quit()

class VehicleDataThread(QThread):
    data_ready = Signal(QRealData)

    def __init__(self):
        super().__init__()

    def run(self):
        q = messaging.q_processing.get_consumer()
        self._is_running = True
        
        while self._is_running and not self.isInterruptionRequested():
            processed_real_data: ProcessedRealData = q.get(100)
            if processed_real_data is None:
                continue
            jpg_image_data: JPGImageData = processed_real_data.original.sensor_fusion.camera
            image_array = imdecode(np.frombuffer(jpg_image_data.jpg, np.uint8), IMREAD_COLOR)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            qvehicle_data = QRealData(processed_real_data)
            qvehicle_data.rgb_qimage = npimage2qimage(image_array)
            qvehicle_data.rgb_updated_qimage = npimage2qimage(processed_real_data.debug_image)
            self.data_ready.emit(qvehicle_data)

    def stop(self):
        self._is_running = False
        self.quit()