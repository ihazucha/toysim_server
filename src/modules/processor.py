from multiprocessing import Process
from time import sleep

from utils.ipc import SPMCQueue
from modules.controller import DualSense

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
        controller = DualSense()
        if not controller.is_connected():
            print(f"[{self.__class__.__name__}] Controller not connected, terminating..")
            return
       
        q_remote = self._q_remote.get_producer()
        while True:
            control_data = controller.get_input()
            q_remote.put(control_data)
            # TODO: artificial sleep to prevent UDP bombardment
            sleep(0.01)
