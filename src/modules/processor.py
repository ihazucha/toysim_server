from multiprocessing import Process
from time import sleep, time_ns
import traceback
import numpy as np
import cv2

from modules.messaging import messaging
from datalink.data import (
    ProcessedData,
    SimData,
    ControlData,
    PurePursuitPIDConfig,
    ImageParams,
    Position,
    Rotation,
)
from modules.controller import DualSense, PurePursuitPID
from modules.path_planning.red_roadmarks import RedRoadmarksPlanner, Camera


from threading import Thread
from enum import Enum


# TODO: figure out more elegant way to update to avoid blocking
# --------------------------------------------------------------
# Solution to implement:
# Process that acts as a proxy for communication between the modules (processes),
# receives and sends messages between them and updates their settings accordingly
# This will be done by registering callbacks that will be called upon message receive from
# a given interface
# Message should be confirmed by the receiver upon delivery with an ACK message

def apply_ui_config(controller: PurePursuitPID):
    q_ui = messaging.q_ui.get_consumer()
    while True:
        config: PurePursuitPIDConfig = q_ui.get()
        controller.set_config(config)
def start_apply_ui_config_thread(controller: PurePursuitPID):
    Thread(target=apply_ui_config, args=[controller], daemon=True).start()


# --------------------------------------------------------------


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


# TODO: precalculate UV (image coordinates) for all points on the road
# TODO: remove None checks - should be empty arrays for more clarity
def draw_debug_data(
    image, planner: RedRoadmarksPlanner, controller: PurePursuitPID
) -> np.ndarray:
    image_copy = image.copy()
    roadmarks = planner.roadmarks_imageframe
    path_uv = [planner.camera.xyz_roadframe_iso88552uv(np.array([*xy, 0])) for xy in planner.path]
    intersections_uv = [
        planner.camera.xyz_roadframe_iso88552uv(np.array([*xy, 0]))
        for xy in controller.pp.filtered_intersections
    ]

    # Roadmark dots
    for u, v in roadmarks:
        cv2.circle(image_copy, (u, v), 5, (0, 255, 0), -1)

    # Planned path
    for i in range(len(path_uv) - 1):
        cv2.line(image_copy, path_uv[i], path_uv[i + 1], (0, 255, 0), 2)

    # PurePursuit goal
    for u, v in intersections_uv:
        cv2.circle(image_copy, (u, v), 5, (255, 255, 255), -1)
        cv2.putText(
            image_copy,
            f"({u}, {v})",
            (u + 10, v - 10),
            cv2.QT_FONT_NORMAL,
            0.5,
            (255, 255, 255),
        )
    return image_copy


class ControllerType(Enum):
    DUALSENSE = "dualsense"
    REDROADMARKS = "redroadmarks"


class Processor(Process):
    def __init__(self, controller_type: ControllerType):
        super().__init__()
        self._controller_type = controller_type

    def _log(self, msg: str):
        print(f"[{self.__class__.__name__}] {msg}")

    def run(self):
        _run = {
            ControllerType.DUALSENSE: self._run_dualsense,
            ControllerType.REDROADMARKS: self._run_redroadmarks,
        }.get(self._controller_type)

        if _run is None:
            self._log(f'Unknown controller "{self._controller_type}", stopping..')
            return

        while True:
            try:
                _run()
            except:
                self._log(f"Controller {self._controller_type} error:")
                traceback.print_exc()
                self._log(f"Attempting restart after 1 second..")
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
        # TODO: obtain data about the camera from the system
        camera = Camera(
            Position(0, 250, 0),
            Rotation(0, -15.05, 0),
            ImageParams(width=640, height=480, fov_deg=90),
        )
        planner = RedRoadmarksPlanner(camera=camera)
        controller = PurePursuitPID(config=PurePursuitPIDConfig())

        q_control = messaging.q_control.get_producer()
        q_simulation = messaging.q_simulation.get_consumer()
        q_processing = messaging.q_processing.get_producer()

        start_apply_ui_config_thread(controller)

        # Sync
        _ = SimData.from_bytes(q_simulation.get())

        while True:
            data: SimData = SimData.from_bytes(q_simulation.get())
            planner.update(rgb_image=data.camera.rgb_image)
            controller.update(path=planner.path, speed=data.vehicle.speed, dt=data.dt)

            c_data = ControlData(controller.timestamp, controller.speed, -controller.steering_angle)
            q_control.put(c_data)
            
            # TODO: should be done by simulation/car
            depth = depth_from_image_center(data.camera.depth_image)
            debug_image = draw_debug_data(data.camera.rgb_image, planner, controller)
            p_data = ProcessedData(begin_timestamp=q_simulation.last_get_timestamp, debug_image=debug_image, depth=depth, original=data)
            q_processing.put(p_data)
