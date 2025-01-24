import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from modules.recorder import RecordReader 
from utils.data import last_record_path

from utils.data import Position, Rotation, SimData
from scipy.interpolate import interp1d


FRAME_BIN_PATH = os.path.join(os.path.dirname(__file__), "../data/sandbox/dataframe.bin")

def write_last_record_first_frame():
    record_path = last_record_path()
    if record_path is None:
        raise ValueError("[Planner] no records found")
    data: list[SimData] = RecordReader.read(record_path=record_path)
    with open(FRAME_BIN_PATH, "wb") as f:
        pickle.dump(data[0], f)

def read_last_record_first_frame():
    with open(FRAME_BIN_PATH, "rb") as f:
        return pickle.load(f)

def get_last_record() -> list[SimData]:
    record_path = last_record_path()
    if record_path is None:
        raise ValueError("[Planner] no records found")
    data: list[SimData] = RecordReader.read(record_path=record_path)
    return data

class PurePursuit:
    param_K_dd = 0.6

    # The above parameters will be used in the Carla simulation
    # The simple simulation in tests/control/control.ipynb does not use these parameters
    def __init__(self, K_dd=param_K_dd, wheel_base=3.1, waypoint_shift=1.4):
        self.K_dd = K_dd
        self.wheel_base = wheel_base
        self.waypoint_shift = waypoint_shift
        self.filtered_intersections = None

    def get_control(self, waypoints, speed):
        # Transform waypoints coordinates such that the frame origin is in the rear wheel
        waypoints[:, 0] += self.waypoint_shift
        look_ahead_distance = np.clip(self.K_dd * speed, 5, 20)

        track_point = self.get_target_point(look_ahead_distance, waypoints)
        if track_point is None:
            return 0

        alpha = np.arctan2(track_point[1], track_point[0])
        steer = np.arctan((2 * self.wheel_base * np.sin(alpha)) / look_ahead_distance)

        # TODO: make copy to prevent redo?
        waypoints[:, 0] -= self.waypoint_shift
        return steer

    # Function from https://stackoverflow.com/a/59582674/2609987
    def circle_line_segment_intersection(
        self, circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9
    ):
        """Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

        :param circle_center: The (x, y) location of the circle center
        :param circle_radius: The radius of the circle
        :param pt1: The (x, y) location of the first point of the segment
        :param pt2: The (x, y) location of the second point of the segment
        :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
        :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
        :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

        Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
        """

        (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
        (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
        dx, dy = (x2 - x1), (y2 - y1)
        dr = (dx**2 + dy**2) ** 0.5
        big_d = x1 * y2 - x2 * y1
        discriminant = circle_radius**2 * dr**2 - big_d**2

        if discriminant < 0:  # No intersection between circle and line
            return []
        else:  # There may be 0, 1, or 2 intersections with the segment
            intersections = [
                (
                    cx
                    + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**0.5) / dr**2,
                    cy + (-big_d * dx + sign * abs(dy) * discriminant**0.5) / dr**2,
                )
                for sign in ((1, -1) if dy < 0 else (-1, 1))
            ]  # This makes sure the order along the segment is correct
            if (
                not full_line
            ):  # If only considering the segment, filter out intersections that do not fall within the segment
                fraction_along_segment = [
                    (xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy
                    for xi, yi in intersections
                ]
                intersections = [
                    pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1
                ]
            if (
                len(intersections) == 2 and abs(discriminant) <= tangent_tol
            ):  # If line is tangent to circle, return just one point (as both intersections have same location)
                return [intersections[0]]
            else:
                return intersections

    def get_target_point(self, lookahead, polyline):
        """Determines the target point for the pure pursuit controller

        Parameters
        ----------
        lookahead : float
            The target point is on a circle of radius `lookahead`
            The circle's center is (0,0)
        poyline: array_like, shape (M,2)
            A list of 2d points that defines a polyline.

        Returns:
        --------
        target_point: numpy array, shape (,2)
            Point with positive x-coordinate where the circle of radius `lookahead`
            and the polyline intersect.
            Return None if there is no such point.
            If there are multiple such points, return the one that the polyline
            visits first.
        """
        intersections = []
        for j in range(len(polyline) - 1):
            pt1 = polyline[j]
            pt2 = polyline[j + 1]
            intersections += self.circle_line_segment_intersection(
                (0, 0), lookahead, pt1, pt2, full_line=False
            )
        filtered = [p for p in intersections if p[0] > 0]
        self.filtered_intersections = filtered
        if len(filtered) == 0:
            return None
        return filtered[0]


class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.last_p = self.i = self.d = 0

    def get_control(self, measurement, set_point, dt):
        p = set_point - measurement
        self.i += p * dt
        self.d = (p - self.last_p) / dt
        self.last_p = p
        return self.Kp * p + self.Ki * self.i + self.Kd * self.d

class ImageParams:
    def __init__(self, width: int, height: int, fov_deg: float):
        self.width = width
        self.height = height
        self.fov_deg = fov_deg


class Camera:
    def __init__(self, position: Position, rotation: Rotation, img_params: ImageParams):
        self.position = position
        self.rotation = rotation
        self.img_params = img_params

        self.M = self.intrinsic_matrix()
        self.M_i = np.linalg.inv(self.M)

        self.R_rc = self.rotation_matrix()
        self.R_cr = self.R_rc.T

        self.T_rc = np.array([self.position.x, self.position.y, self.position.z])
        self.T_cr = self.T_rc * -1.0

        self.H_cr = np.eye(4)
        self.H_cr[:3, :3] = self.R_cr
        self.H_cr[:3, 3] = self.T_cr
        self.H_rc = np.linalg.inv(self.H_cr)

        self.road_normal_camera_frame = self.R_rc @ np.array([0, 1, 0])

        self.calc_or_load_xy_roadframe_iso8855()

    def intrinsic_matrix(self) -> np.ndarray:
        fov = np.deg2rad(self.img_params.fov_deg)
        alpha = (self.img_params.width / 2.0) / np.tan(fov / 2.0)
        return np.array(
            [
                [alpha, 0, self.img_params.width / 2],
                [0, alpha, self.img_params.height / 2],
                [0, 0, 1],
            ]
        )

    def rotation_matrix(self) -> np.ndarray:
        roll = np.deg2rad(self.rotation.roll)
        pitch = np.deg2rad(self.rotation.pitch)
        yaw = np.deg2rad(self.rotation.yaw)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array(
            [
                [cr * cy + sp * sr + sy, cr * sp * sy - cy * sr, -cp * sy],
                [cp * sr, cp * cr, sp],
                [cr * sy - cy * sp * sr, -cr * cy * sp - sr * sy, cp * cy],
            ]
        )

    def calc_or_load_xy_roadframe_iso8855(self):
        file_name = f"cache_xy_roadframe_iso8855_{self.img_params.width}x{self.img_params.height}_fov{self.img_params.fov_deg}.npy"
        try:
            self.xy_roadframe_iso8855 = np.load(file_name)
            print(f"Loaded cached data from {file_name}")
        except FileNotFoundError:
            self.xy_roadframe_iso8855 = self.image2xy_roadframe_iso8855()
            np.save(file_name, self.xy_roadframe_iso8855)
            print(f"Saved cache data to {file_name}")

    def cam2road(self, vec: np.ndarray) -> np.ndarray:
        return (self.R_cr @ vec) + self.T_cr

    def road2cam(self, vec: np.ndarray) -> np.ndarray:
        return self.R_rc @ (vec + self.T_rc)

    def uv2xyz_camframe(self, u: int, v: int) -> np.ndarray:
        image_point = [u, v, 1]
        direction_vector = self.M_i @ image_point
        scaling_factor = self.position.y / (self.road_normal_camera_frame @ direction_vector)
        xyz_c = scaling_factor * direction_vector
        return xyz_c

    def xyz_camframe2uv(self, xyz: np.ndarray) -> np.ndarray:
        uvw = self.M @ xyz
        uvw /= uvw[2]
        u, v = int(uvw[0]), int(uvw[1])
        return np.array([u, v])

    def uv2xyz_roadframe(self, u: int, v: int) -> np.ndarray:
        xyz_c = self.uv2xyz_camframe(u, v)
        return self.cam2road(xyz_c)

    def uv2xyz_roadframe_iso8855(self, u: int, v: int) -> np.ndarray:
        x, y, z = self.uv2xyz_roadframe(u, v)
        return np.array([z, -x, -y])

    def xyz_roadframe2uv(self, xyz):
        xyz_camframe = self.road2cam(xyz)
        return self.xyz_camframe2uv(xyz_camframe)

    def xyz_roadframe_iso88552uv(self, xyz: np.ndarray) -> np.ndarray:
        x, y, z = -xyz[1], -xyz[2], xyz[0]
        return self.xyz_roadframe2uv(np.array([x, y, z]))

    def image2xy_roadframe_iso8855(self) -> np.ndarray:
        # TODO: make fast
        xy = np.zeros((self.img_params.width, self.img_params.height, 2))
        for u in range(self.img_params.width):
            for v in range(self.img_params.height):
                x, y, _ = self.uv2xyz_roadframe_iso8855(u, v)
                xy[u, v] = x, y
        return xy

    def v_threshold_by_distance(self, distance: float):
        road_point_road_frame = np.array([0, 0, distance, 1])
        road_point_cam_frame = self.H_rc @ road_point_road_frame
        uv_vec = self.M @ road_point_cam_frame[:3]
        uv_vec /= uv_vec[2]
        cut_v = uv_vec[1]
        return cut_v


class RedRoadmarksPathPlanner:
    def __init__(self, camera: Camera):
        self.camera = camera

        # Intermediate calculations for easier debugging
        self.bgr_image = None
        self.bgr_filtered = None
        self.roadmarks_imageframe = None
        self.roadmarks = None
        self.roadmarks_interp = None
        self.path = None

    def plan(self, rgb_image: np.ndarray) -> np.ndarray:
        self.bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        self.bgr_filtered = RedRoadmarksPathPlanner.red_color_thresh_filter(self.bgr_image)
        self.roadmarks_imageframe = self.find_roadmarks(self.bgr_filtered)
        self.roadmarks = np.array(
            [self.camera.xy_roadframe_iso8855[uv] for uv in self.roadmarks_imageframe]
        )
        if self.roadmarks.size < 2:
            # TODO: think about better approach
            return self.roadmarks
        self.roadmarks_interp = RedRoadmarksPathPlanner.interpolate_roadmarks(self.roadmarks)
        self.path = RedRoadmarksPathPlanner.polyline_fit_path(self.roadmarks_interp)
        return self.path

    @staticmethod
    def red_color_thresh_filter(bgr_image):
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # Red in HSV space is depicted by Hue around 360 deg
        # which is around 179 (max value in OpenCV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([6, 255, 255])
        lower_red2 = np.array([173, 50, 50])
        upper_red2 = np.array([189, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask = mask1 + mask2

        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        hsv_filtered = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
        return cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR)

    def find_roadmarks(
        self, bgr_filtered: np.ndarray, min_area: float = 10, max_roadmarks: int = 6
    ) -> list:
        # TODO: find better way to detect valid roadmarks
        gray = cv2.cvtColor(bgr_filtered, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roadmark_centers = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            xc, yc = (x + w // 2, y + h // 2)
            if (xc < (self.camera.img_params.width * 0.2)) or (
                xc > (self.camera.img_params.width * 0.8)
            ):
                continue
            roadmark_centers.append((xc, yc))
        return roadmark_centers[:max_roadmarks] if max_roadmarks > 0 else roadmark_centers

    @staticmethod
    def interpolate_roadmarks(roadmarks: np.ndarray) -> np.ndarray:
        f = interp1d(roadmarks[:, 0], roadmarks[:, 1], kind="linear")
        xs = np.linspace(roadmarks[0, 0], roadmarks[-1, 0], num=5)
        ys = f(xs)
        return np.column_stack((xs, ys))

    @staticmethod
    def polyline_fit_path(roadmarks: np.ndarray) -> np.ndarray:
        coeffs = np.polyfit(x=roadmarks[:, 0], y=roadmarks[:, 1], deg=3)
        f = np.poly1d(coeffs)
        xs = np.arange(roadmarks[0][0], roadmarks[-1][0], 50)
        ys = f(xs)
        return np.column_stack((xs, ys))


class Controller:
    cm2m = 1e2
    dt = 1.0 / 30.0

    def __init__(self, pure_pursuit=PurePursuit(), pid=PIDController(Kp=2, Ki=0, Kd=0)):
        self.pure_pursuit = pure_pursuit
        self.pid = pid

    def get_inputs(self, path: np.ndarray, speed_cmps: float):
        path_m = path / Controller.cm2m
        speed_ms = speed_cmps / Controller.cm2m
        self.set_speed = self.pid.get_control(measurement=speed_ms, set_point=1.0, dt=Controller.dt)
        self.set_steering_angle = np.rad2deg(self.pure_pursuit.get_control(path_m, speed_ms))
        return (self.set_speed, self.set_steering_angle)


def main():
    records: list[SimData] = get_last_record()
    camera = Camera(
        Position(0, 251, 0), Rotation(0, -14.33, 0), ImageParams(width=640, height=480, fov_deg=90)
    )

    path_planner = RedRoadmarksPathPlanner(camera=camera)
    path_visualiser = RedRoadmarksPathPlannerVisualiser(path_planner)

    controller = Controller()
    controller_visualiser = ControllerVisualiser(controller)

    for data in records:
        path = path_planner.plan(data.camera_data.rgb_image)
        if path.size < 1:
            print("[Control] unable to detect path")
            continue

        intersections = controller.pure_pursuit.filtered_intersections
        path_visualiser.plot(intersections=intersections)

        inputs = controller.get_inputs(path, data.vehicle_data.speed)
        controller_visualiser.save()
    controller_visualiser.plot()


# Visualisers
# ----------------------------------------------------------------------------------------------------


class RedRoadmarksPathPlannerVisualiser:
    def __init__(self, planner: RedRoadmarksPathPlanner):
        self.planner = planner

    def plot(self, dt: float = 1.0 / 30.0, intersections=None):
        for i in range(len(self.planner.path) - 1):
            # print(self.planner.path)
            xyz_i = np.array([*self.planner.path[i], 0])
            xyz_ii = np.array([*self.planner.path[i + 1], 0])
            uv_i = self.planner.camera.xyz_roadframe_iso88552uv(xyz_i)
            uv_ii = self.planner.camera.xyz_roadframe_iso88552uv(xyz_ii)
            cv2.line(self.planner.bgr_filtered, uv_i, uv_ii, (255, 0, 0), 2)

        self.__class__.draw_roadmarks(self.planner.bgr_filtered, self.planner.roadmarks_imageframe)
        if intersections is not None:
            self.draw_intersections(self.planner.bgr_filtered, intersections)

        combined_image = np.hstack((self.planner.bgr_image, self.planner.bgr_filtered))
        cv2.imshow("Combined Image", combined_image)
        if cv2.waitKey(33) & 0xFF == ord("q"):
            exit()

    @staticmethod
    def draw_roadmarks(img, roadmarks):
        for cc in roadmarks:
            cv2.circle(img, cc, 2, (0, 255, 0), -1)
            cv2.putText(
                img,
                f"({cc[0]}, {cc[1]})",
                (cc[0] + 10, cc[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    def draw_intersections(self, img, intersections):
        intersections_uv = []
        if intersections is not None:
            for x, y in intersections:
                p_extended = np.array([x * 100, y * 100, 1])
                uv = self.planner.camera.xyz_roadframe_iso88552uv(p_extended)
                intersections_uv.append(uv)

        for u, v in intersections_uv:
            cv2.circle(img, (u, v), 5, (255, 255, 255), -1)
            cv2.putText(
                img,
                f"({u}, {v})",
                (u + 10, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )


class ControllerVisualiser:
    def __init__(self, controller: Controller):
        self.controller = controller
        self.steering_setpoints = []
        self.speed_setpoints = []

    def save(self):
        self.speed_setpoints.append(self.controller.set_speed)
        self.steering_setpoints.append(self.controller.set_steering_angle)

    def plot(self):
        # Plotting speed and steering setpoints
        plt.figure(figsize=(12, 6))

        # Speed setpoints plot
        plt.subplot(1, 2, 1)
        plt.plot(self.speed_setpoints, label="Speed Setpoints")
        plt.xlabel("Time Step")
        plt.ylabel("Speed (m/s)")
        plt.title("Speed Setpoints Over Time")
        plt.legend()
        plt.grid(True)

        # Steering setpoints plot
        plt.subplot(1, 2, 2)
        plt.plot(self.steering_setpoints, label="Steering Setpoints", color="orange")
        plt.xlabel("Time Step")
        plt.ylabel("Steering Angle (degrees)")
        plt.title("Steering Setpoints Over Time")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def pixel_target_tool(cam: Camera, bgr, transformed_depth, depth):
    pixel_target = [cam.img_width // 2, cam.img_height // 2]

    def update_pixel_target(val):
        u = cv2.getTrackbarPos("U", "depth")
        v = cv2.getTrackbarPos("V", "depth")
        target_camframe = cam.uv2xyz_camframe(u, v)
        dist = np.linalg.norm(target_camframe)
        target_roadframe = cam.uv2xyz_roadframe(u, v)
        print(
            f"UV: ({u}, {v}) "
            f"XYZc: {[int(x) for x in target_camframe]} "
            f"XYZr: {[int(x) for x in target_roadframe]} "
            f"D_gt: {depth[v,u]} "
            f"D_gt_t: {transformed_depth[v][u]} "
            f"D: {dist:.1f}"
        )

        # Draw cross point on the image
        img_copy = bgr.copy()
        cv2.drawMarker(
            img_copy, (u, v), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1
        )
        cv2.imshow("depth", img_copy)

    cv2.namedWindow("depth")
    cv2.createTrackbar("U", "depth", pixel_target[0], cam.img_width - 1, update_pixel_target)
    cv2.createTrackbar("V", "depth", pixel_target[1], cam.img_height - 1, update_pixel_target)
    cv2.imshow("depth", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def transform_depth_to_camera_center(depth, cam):
    transformed_depth = np.zeros_like(depth, dtype=np.float32)
    for v in range(cam.img_height):
        for u in range(cam.img_width):
            depth_value = depth[v, u]
            if depth_value > 0:
                # Convert depth value to 3D point in camera frame
                xyz_cam = cam.uv2xyz_camframe(u, v)
                xyz_cam[2] = depth_value  # Update the Z value with the depth value

                # Calculate the distance from the camera center to the object
                distance_to_camera_center = np.linalg.norm(xyz_cam)
                transformed_depth[v, u] = distance_to_camera_center
    return transformed_depth


if __name__ == "__main__":
    main()
