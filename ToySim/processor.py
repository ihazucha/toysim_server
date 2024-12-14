from multiprocessing import Process
from time import time, sleep

from ToySim.ipc import SPMCQueue
from ToySim.data import RemoteControlData

class Processor(Process):
    def __init__(
        self,
        q_image: SPMCQueue,
        q_sensor: SPMCQueue,
        q_control: SPMCQueue,
    ):
        super().__init__()
        self._q_image = q_image
        self._q_sensor = q_sensor
        self._q_control = q_control

    def run(self):
        q_control = self._q_control.get_producer()
        while True:
            control_data = RemoteControlData(time(), 0.0, 0.0)
            q_control.put(control_data.to_bytes())
            # TODO: artificial sleep to prevent UDP bombardment
            sleep(0.05)