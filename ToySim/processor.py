from multiprocessing import Process
from time import time, sleep

from ToySim.ipc import SharedBuffer
from ToySim.data import RemoteControlData

class Processor(Process):
    def __init__(
        self,
        q_image: SharedBuffer.Reader,
        q_sensor: SharedBuffer.Reader,
        q_control: SharedBuffer.Writer,
    ):
        super().__init__()
        self._q_image = q_image
        self._q_sensor = q_sensor
        self._q_control = q_control

    def run(self):
        while True:
            control_data = RemoteControlData(time(), 0.0, 0.0)
            self._q_control.write(control_data.to_bytes())
            # TODO: artificial sleep to prevent UDP bombardment
            sleep(0.05)