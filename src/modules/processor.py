import cv2
import numpy as np
from multiprocessing import Process
from time import sleep, time_ns
import traceback

from projection import ImageParams
from utils.ipc import SPMCQueue
from utils.data import Position, Rotation, SimData, RemoteControlData
from modules.controller import DualSense, PurePursuitPIDController
from modules.path_planning.red_roadmarks import RedRoadmarksPathPlanner, Camera


class Processor(Process):
    def __init__(
        self,
        q_image: SPMCQueue,
        q_sensor: SPMCQueue,
        q_remote: SPMCQueue,
        q_simulation: SPMCQueue,
        controller: str,
    ):
        super().__init__()
        self._q_image = q_image
        self._q_sensor = q_sensor
        self._q_remote = q_remote
        self._q_simulation = q_simulation
        self._controller = controller

    def run(self):
        while True:
            try:
                if self._controller == "dualsense":
                    self._run_dualsense()
                elif self._controller == "redroadmarks":
                    self._run_redroadmarks()
            except:
                print(f"[{self.__class__.__name__}] Controller {self._controller} error:")
                traceback.print_exc()
                print(f"[{self.__class__.__name__}] Attempting restart after 1 second..")
                sleep(1)

    def _run_dualsense(self):
        controller = DualSense()
        if not controller.is_connected():
            print(f"[{self.__class__.__name__}] {controller.__class__.__name__} unable to connect")
            return

        q_remote = self._q_remote.get_producer()
        while True:
            control_data = controller.get_input()
            q_remote.put(control_data)
            # TODO: prevent UDP flood
            sleep(0.01)

    def _run_redroadmarks(self):
        # TODO: this camera only works for simulation
        camera = Camera(
            Position(0, 251, 0),
            Rotation(0, -14.33, 0),
            ImageParams(width=640, height=480, fov_deg=90),
        )
        planner = RedRoadmarksPathPlanner(camera)
        controller = PurePursuitPIDController()

        q_remote = self._q_remote.get_producer()
        q_simulation = self._q_simulation.get_consumer()
        while True:
            sim_data: SimData = SimData.from_bytes(q_simulation.get())
            path = planner.plan(sim_data.camera_data.rgb_image)
            set_speed, set_steering_angle = controller.get_inputs(
                path=path, speed_cmps=sim_data.vehicle_data.speed, dt=sim_data.dt
            )
            remote_control_data = RemoteControlData(
                timestamp=time_ns(), set_speed=set_speed, set_steering_angle=-set_steering_angle
            )
            q_remote.put(remote_control_data)
