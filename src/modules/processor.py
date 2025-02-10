from multiprocessing import Process
from time import sleep, time_ns
import traceback
import numpy as np
import cv2

from utils.ipc import messaging
from utils.data import ControllerData, SimData, RemoteControlData, UIConfigData
from modules.controller import DualSense, PurePursuitPIDController
from modules.path_planning.red_roadmarks import RedRoadmarksPathPlanner

from threading import Thread
from enum import Enum


class ControllerType(Enum):
    DUALSENSE = "dualsense"
    REDROADMARKS = "redroadmarks"


class Processor(Process):

    def __init__(self, controller: ControllerType):
        super().__init__()
        self._controller = controller
        self._last_controller_config = UIConfigData()

    def run(self):
        while True:
            try:
                self._run_controller()
            except:
                self._error_recovery()

    def _run_controller(self):
        routines = {
            ControllerType.DUALSENSE: self._run_dualsense,
            ControllerType.REDROADMARKS: self._run_redroadmarks,
        }
        routine = routines.get(self._controller)
        assert routine, print(f"[{self.__class__.__name__}] Unknown controller: {self._controller}")
        routine()

    def _error_recovery(self, sleep_time: int = 1):
        print(f"[{self.__class__.__name__}] Controller {self._controller} error:")
        traceback.print_exc()
        print(f"[{self.__class__.__name__}] Attempting restart after {sleep_time} second...")
        sleep(sleep_time)

    def _run_dualsense(self):
        controller = DualSense()
        if not controller.is_connected():
            raise ValueError("[DualSense] unable to connnect")

        q_control = self._q_control.get_producer()
        while True:
            control_data = controller.get_input()
            q_control.put(control_data)
            sleep(0.02)

    def _run_redroadmarks(self):
        planner = RedRoadmarksPathPlanner()
        controller = PurePursuitPIDController()
        controller.update_config(self._last_controller_config)

        q_control = messaging.q_control.get_producer()
        q_simulation = messaging.q_simulation.get_consumer()
        q_processing = messaging.q_processing.get_producer()

        # TODO: figure out more elegant way to update to avoid blocking
        # --------------------------------------------------------------
        # Solution to implement:
        # Process that acts as a proxy for communication between the modules (processes),
        # receives and sends messages between them and updates their settings accordingly
        # This will be done by registering callbacks that will be called upon message receive from
        # a given interface
        # Message should be confirmed by the receiver upon delivery with an ACK message
        def ui_update():
            q_ui = messaging.q_ui.get_consumer()
            while True:
                new_config = q_ui.get()
                self._last_controller_config = new_config
                controller.update_config(new_config)

        t_ui_update = Thread(target=ui_update, daemon=True)
        t_ui_update.start()
        # --------------------------------------------------------------

        # TODO: precalculate UV (image coordinates) for all points on the road
        def draw_data_to_image(image, path, intersections, roadmarks) -> np.ndarray:
            image_copy = image.copy()
            path_uv = [planner.camera.xyz_roadframe_iso88552uv(np.array([*xy, 0])) for xy in path]
            intersections_uv = (
                [
                    planner.camera.xyz_roadframe_iso88552uv(np.array([*xy, 0]))
                    for xy in intersections
                ]
                if intersections
                else None
            )
            for i in range(len(path_uv) - 1):
                uv1 = path_uv[i]
                uv2 = path_uv[i + 1]
                cv2.line(image_copy, uv1, uv2, (0, 255, 0), 2)

            if roadmarks:
                for u, v in roadmarks:
                    cv2.circle(image_copy, (u, v), 5, (0, 255, 0), -1)

            if intersections_uv:
                for u, v in intersections_uv:
                    cv2.circle(image_copy, (u, v), 5, (255, 255, 255), -1)
                    cv2.putText(
                        image_copy,
                        f"({u}, {v})",
                        (u + 10, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
            return image_copy

        while True:
            sim_data: SimData = SimData.from_bytes(q_simulation.get())
            path = planner.plan(sim_data.camera_data.rgb_image)
            set_speed, set_steering_angle = controller.get_inputs(
                path=path, speed_cmps=sim_data.vehicle_data.speed, dt=sim_data.dt
            )

            updated_image = draw_data_to_image(
                sim_data.camera_data.rgb_image,
                path,
                controller.pure_pursuit.filtered_intersections,
                planner.roadmarks_imageframe,
            )
            controller_data = ControllerData(image=updated_image)
            q_processing.put(controller_data)

            remote_control_data = RemoteControlData(
                timestamp=time_ns(), set_speed=set_speed, set_steering_angle=-set_steering_angle
            )
            q_control.put(remote_control_data)
