from multiprocessing import Process
from time import sleep, time_ns
import traceback
import numpy as np
import cv2

from datalink.ipc import messaging
from datalink.data import (
    ProcessedData,
    SimData,
    ControlData,
    UIConfigData,
    ImageParams,
    Position,
    Rotation,
)
from modules.controller import DualSense, PurePursuitPIDController
from modules.path_planning.red_roadmarks import RedRoadmarksPathPlanner, Camera


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
        controller_run_method = {
            ControllerType.DUALSENSE: self._run_dualsense,
            ControllerType.REDROADMARKS: self._run_redroadmarks,
        }.get(self._controller)

        if controller_run_method is None:
            print(f"[{self.__class__.__name__}] Unknown controller: {self._controller}, stopping.")
            return

        while True:
            try:
                controller_run_method()
            except:
                print(f"[{self.__class__.__name__}] Controller {self._controller} error:")
                traceback.print_exc()
                print(f"[{self.__class__.__name__}] Attempting restart after 1 second..")
                sleep(1)

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
        camera = Camera(
            Position(0, 250, 0),
            Rotation(0, -15.05, 0),
            ImageParams(width=640, height=480, fov_deg=90),
        )
        planner = RedRoadmarksPathPlanner(camera=camera)
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
                print(new_config)
                controller.update_config(new_config)

        t_ui_update = Thread(target=ui_update, daemon=True)
        t_ui_update.start()
        # --------------------------------------------------------------

        # TODO: precalculate UV (image coordinates) for all points on the road
        # TODO: remove None checks - should be empty arrays for more clarity
        def draw_debug_data(image, path, planner, controller) -> np.ndarray:
            image_copy = image.copy()
            intersections = controller.pure_pursuit.filtered_intersections
            roadmarks = planner.roadmarks_imageframe
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

        # TODO: clean up to some other place
        def compute_radial_dist(width: int, height: int):
            radial_dist = np.zeros((height, width), dtype=np.float32)
            h, w = radial_dist.shape
            center_x, center_y = w // 2, h // 2
            for y in range(h):
                for x in range(w):
                    dx = x - center_x
                    dy = y - center_y
                    radial_dist[y, x] = dx**2 + dy**2
            return radial_dist

        RADIAL_DIST = compute_radial_dist(width=640, height=480)

        def depth_from_image_center(depth: np.ndarray):
            """The depth values returned by the UE5 simulation contain scene depth
            which is measured from each individual pixel to the point it is capturing.
            This method calculates distance from the image center, which corresponds
            to the camera location inside the simulation
            """
            return np.sqrt(RADIAL_DIST + depth.astype(np.float32) ** 2)

        # Wait for first data
        # _ = SimData.from_bytes(q_simulation.get())
        
        t_last = time_ns()
        while True:
            data: SimData = SimData.from_bytes(q_simulation.get())
            # print(f"{data.dt}")
            path = planner.plan(data.camera.rgb_image)
            controller.update(path=path, v=data.vehicle.speed, dt=data.dt)

            c_data = ControlData(timestamp=controller.timestamp, v=controller.v, sa=-controller.sa)
            q_control.put(c_data)

            # TODO: should be done by simulation/car
            depth = depth_from_image_center(data.camera.depth_image)
            debug_image = draw_debug_data(data.camera.rgb_image, path, planner, controller)
            t = time_ns()
            p_data = ProcessedData(dt=(t-t_last)/1e6, debug_image=debug_image, depth=depth, original=data)
            t_last = t
            q_processing.put(p_data)

