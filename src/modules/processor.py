from multiprocessing import Process
from time import sleep, time_ns
import traceback
import numpy as np
import cv2

from projection import ImageParams
from utils.ipc import messaging
from utils.data import ControllerData, Position, Rotation, SimData, RemoteControlData
from modules.controller import DualSense, PurePursuitPIDController
from modules.path_planning.red_roadmarks import RedRoadmarksPathPlanner, Camera

from threading import Thread

class Processor(Process):
    def __init__(self, controller: str):
        super().__init__()
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
            raise ValueError("[DualSense] unable to connnect")

        q_remote = self._q_remote.get_producer()
        while True:
            control_data = controller.get_input()
            q_remote.put(control_data)
            sleep(0.02)

    def _run_redroadmarks(self):
        # TODO: this camera only works for simulation
        camera = Camera(
            Position(0, 251, 0),
            Rotation(0, -14.33, 0),
            ImageParams(width=640, height=480, fov_deg=90),
        )
        planner = RedRoadmarksPathPlanner(camera)
        controller = PurePursuitPIDController()

        q_remote = messaging.q_remote.get_producer()
        q_simulation = messaging.q_simulation.get_consumer()
        q_processing = messaging.q_processing.get_producer()

        # TODO: figure out more elegant way to update to avoid blocking
        # --------------------------------------------------------------
        def ui_update():
            q_ui = messaging.q_ui.get_consumer()
            while True:
                new_config = q_ui.get()
                # print(str(new_config))
                controller.update_config(new_config)

        t_ui_update = Thread(target=ui_update, daemon=True)
        t_ui_update.start()
        # --------------------------------------------------------------

        def draw_data_to_image(image, path, intersections) -> np.ndarray:
            image_copy = image.copy()
            path_uv = [planner.camera.xyz_roadframe_iso88552uv(np.array([*xy, 0])) for xy in path]
            intersections_uv = [planner.camera.xyz_roadframe_iso88552uv(np.array([*xy, 0])) for xy in intersections] if intersections else None
            for i in range(len(path_uv) - 1):
                uv1 = path_uv[i]
                uv2 = path_uv[i + 1]
                cv2.line(image_copy, uv1, uv2, (0, 255, 0), 2)
                
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
            
            updated_image = draw_data_to_image(sim_data.camera_data.rgb_image, path, controller.pure_pursuit.filtered_intersections)
            controller_data = ControllerData(image=updated_image)
            q_processing.put(controller_data)
            
            remote_control_data = RemoteControlData(
                timestamp=time_ns(), set_speed=set_speed, set_steering_angle=-set_steering_angle
            )
            q_remote.put(remote_control_data)
