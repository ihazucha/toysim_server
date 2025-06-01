from time import sleep, time_ns

from multiprocessing import Process
import traceback
import numpy as np
import cv2

from modules.messaging import messaging
from datalink.data import (
    ProcessedSimData,
    ProcessedRealData,
    RoadmarksData,
    PurePursuitConfig,
    RealData,
    SimData,
    ControlData,
    Pose,
    Position,
    Rotation,
)
from modules.controller import DualSense, PurePursuitController
from modules.path_planning.roadmarks import (
    RoadmarksPlanner,
    RoadmarksPlannerConfig,
    Camera,
    HSVColorFilter,
    rpi_v2_intrinsic_matrix_from_fov,
    unreal_engine_intrinsic_matrix,
)

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


def apply_ui_config(controller: PurePursuitConfig):
    q_ui = messaging.q_ui.get_consumer()
    while True:
        config: PurePursuitConfig = q_ui.get()
        controller.set_config(config)


def start_apply_ui_config_thread(controller: PurePursuitController):
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


# # TODO: precalculate UV (image coordinates) for all points on the road
# # TODO: remove None checks - should be empty arrays for more clarity
def draw_debug_data(image, planner: RoadmarksPlanner, controller: PurePursuitController) -> np.ndarray:
    image_copy = image.copy()

    # Roadmark dots
    for u, v in planner.roadmarks_imgframe:
        cv2.circle(image_copy, (u, v), 5, (23, 155, 93), -1)

    path_imgframe = [
        planner.camera.xyzw_roadframe2uv(np.array([*xy, 0, 1]))[:2] for xy in planner.path_roadframe
    ]
    # Planned path
    for i in range(len(path_imgframe) - 1):
        cv2.line(image_copy, path_imgframe[i], path_imgframe[i + 1], (23, 155, 93), 2)

    # PurePursuit goal
    if controller.pp.track_point:
        u, v = planner.camera.xyzw_roadframe2uv(np.array([*controller.pp.track_point, 0, 1]))
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
    ROADMARKS = "roadmarks"
    REAL_ROADMARKS = "real_roadmarks"



def depth_from_image_center(radial_dist, depth: np.ndarray):
    """The depth values returned by the UE5 simulation contain scene depth
    which is measured from each individual pixel to the point it is capturing.
    This method calculates distance from the image center, which corresponds
    to the camera location inside the simulation
    """
    return np.sqrt(radial_dist + depth.astype(np.float32) ** 2)


class Processor(Process):
    def __init__(self, controller_type: ControllerType):
        super().__init__()
        self._controller_type = controller_type
        self._run = {
            ControllerType.DUALSENSE: self._run_dualsense,
            ControllerType.ROADMARKS: self._run_roadmarks,
            ControllerType.REAL_ROADMARKS: self._run_real_roadmarks,
        }.get(self._controller_type)
        assert self._run, f'Unknown controller "{self._controller_type}", stopping..'

    def run(self):
        while True:
            try:
                self._run()
            except:
                print(f"Controller {self._controller_type} error:")
                traceback.print_exc()
                print(f"Attempting restart after 1 second..")
                sleep(1)

    def _run_dualsense(self):
        dualsense = DualSense()
        dualsense.connect()
        if not dualsense.is_alive():
            raise ValueError("Unable to connnect to DualSense controller")

        q_control = messaging.q_control.get_producer()
        while True:
            dualsense.update()
            if not dualsense.data_ready:
                raise ValueError("Invalid reading")
            ctrls = ControlData(dualsense.timestamp, dualsense.v, dualsense.sa)
            q_control.put(ctrls)
            sleep(0.02)

    def _run_roadmarks(self):
        # TODO: obtain data about the camera from the system
        image_shape = (640, 480)
        # TODO: adjust FOV in the simulator to match the RPi camera
        camera = Camera(
            pose=Pose(Position(0, 250, 0), Rotation(0, -15.5, 0)),
            image_shape=image_shape,
            intrinsic_matrix=unreal_engine_intrinsic_matrix(image_shape=image_shape, fov_deg=60),
        )
        planner = RoadmarksPlanner(
            camera=camera,
            filter=HSVColorFilter.new_red(),
            config=RoadmarksPlannerConfig(roadmark_min_area=11, roadmark_max_count=6, roadmark_max_distance=100000),
        )
        
        controller_config = PurePursuitConfig.new_simulation()
        controller = PurePursuitController(config=controller_config)
        
        q_simulation = messaging.q_sim.get_consumer()
        q_processing = messaging.q_sim_processing.get_producer()
        q_control = messaging.q_control.get_producer()

        start_apply_ui_config_thread(controller)

        RADIAL_DIST = compute_radial_dist(width=640, height=480)

        # Sync
        _ = SimData.from_bytes(q_simulation.get())

        while True:
            data: SimData = SimData.from_bytes(q_simulation.get())
            img_bgr = cv2.cvtColor(data.camera.rgb_image, cv2.COLOR_RGB2BGR)
            planner.update(img=img_bgr)
            controller.update(path=planner.path_roadframe, speed=data.vehicle.speed)
            c_data = ControlData(controller.timestamp, 0.05, -controller.steering_angle)
            q_control.put(c_data)

            # TODO: should be done by simulation/car
            depth = depth_from_image_center(RADIAL_DIST, data.camera.depth_image)
            debug_image = draw_debug_data(data.camera.rgb_image, planner, controller)
            rm_data = RoadmarksData(roadmarks=planner.roadmarks_roadframe, path=planner.path_roadframe)
            p_data = ProcessedSimData(
                begin_timestamp=q_simulation.last_get_timestamp,
                control_data=c_data,
                debug_image=debug_image,
                depth=depth,
                roadmarks_data=rm_data,
                original=data,
            )
            q_processing.put(p_data)

    def _run_real_roadmarks(self):
        # TODO: obtain data about the camera from the system
        image_shape = (3280 // 4, 2464 // 4)
        camera = Camera(
            pose=Pose(position=Position(0, 0.125, 0), rotation=Rotation(0, 0.01, 0)),
            image_shape=image_shape,
            intrinsic_matrix=rpi_v2_intrinsic_matrix_from_fov(image_shape=image_shape),
        )
        config = RoadmarksPlannerConfig(roadmark_min_area=50, roadmark_max_count=6)
        planner = RoadmarksPlanner(camera=camera, filter=HSVColorFilter.new_bright(), config=config)

        controller_config = PurePursuitConfig.new_alamak()
        controller = PurePursuitController(config=controller_config)
        # remote_controller = DualSense()

        q_control = messaging.q_control.get_producer()
        q_real = messaging.q_real.get_consumer()
        q_processing = messaging.q_real_processing.get_producer()

        # Sync
        _ = RealData.from_bytes(q_real.get())
        # TODO: the controller lib can hang trying to connect, init in separate thread
        # remote_controller.connect()

        while True:
            data: RealData = RealData.from_bytes(q_real.get())
            img = cv2.imdecode(
                np.frombuffer(data.sensor_fusion.camera.jpg, np.uint8), cv2.IMREAD_COLOR
            )
            img = camera.undistort_image(img)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            planner.update(img=img)
            controller.update(path=planner.path_roadframe, speed=data.sensor_fusion.avg_speed)

            speed = 0.0
            servo_corrected_steering_angle = controller.steering_angle * 3
            # print(servo_corrected_steering_angle)
            c_data = ControlData(controller.timestamp, speed, servo_corrected_steering_angle)

            #if remote_controller.update():
            #    if remote_controller.is_v_nonzero():
            #        c_data.speed = remote_controller.v
            #    if remote_controller.is_sa_nonzero():
            #        c_data.steering_angle = -remote_controller.sa

            q_control.put(c_data)

            debug_image = draw_debug_data(img_rgb, planner, controller)

            rm_data = RoadmarksData(
                roadmarks=planner.roadmarks_roadframe, path=planner.path_roadframe
            )
            p_data = ProcessedRealData(
                begin_timestamp=q_real.last_get_timestamp,
                control_data=c_data,
                debug_image=debug_image,
                roadmarks_data=rm_data,
                original=data,
            )
            q_processing.put(p_data)
