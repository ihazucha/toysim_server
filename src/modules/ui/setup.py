import numpy as np
import cv2

from PySide6.QtCore import QThread, Signal

from datalink.ipc import SPMCQueue
from utils.env import Environment, ENV
from datalink.data import JPGImageData, SimData


class RendererUISetup(QThread):
    ui_setup_data_ready = Signal(tuple)

    def __init__(
        self, q_image: SPMCQueue, q_sensor: SPMCQueue, q_control: SPMCQueue, q_simulation: SPMCQueue
    ):
        super().__init__()
        self._q_image = q_image
        self._q_sensor = q_sensor
        self._q_control = q_control
        self._q_simulation = q_simulation

    def run(self):
        setup_data: tuple | None = None
        if ENV == Environment.VEHICLE:
            q_image = self._q_image.get_consumer()
            jpg_image_data: JPGImageData = q_image.get()
            image_array = cv2.imdecode(
                np.frombuffer(jpg_image_data.jpg, np.uint8), cv2.IMREAD_COLOR
            )
            height, width, _ = image_array.shape
            setup_data = (width, height)
        elif ENV == Environment.SIM:
            q_simulation = self._q_simulation.get_consumer()
            sim_data_bytes = q_simulation.get()
            sim_data: SimData = SimData.from_bytes(sim_data_bytes)
            height, width, _ = sim_data.camera.rgb_image.shape
            setup_data = (width, height)
        else:
            raise NotImplementedError()
        self.ui_setup_data_ready.emit(setup_data)
