from multiprocessing import Process
from time import time, sleep

from ToySim.utils.ipc import SPMCQueue  # type: ignore
from ToySim.data import RemoteControlData


class Processor(Process):
    def __init__(
        self,
        q_image: SPMCQueue,
        q_sensor: SPMCQueue,
        q_remote: SPMCQueue,
    ):
        super().__init__()
        self._q_image = q_image
        self._q_sensor = q_sensor
        self._q_remote = q_remote

    def run(self):
        q_remote = self._q_remote.get_producer()
        while True:
            control_data = RemoteControlData(time(), 0.0, 0.0)
            q_remote.put(control_data)
            # TODO: artificial sleep to prevent UDP bombardment
            sleep(0.05)
