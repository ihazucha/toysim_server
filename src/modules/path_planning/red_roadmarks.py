import numpy as np
import cv2

from scipy.interpolate import interp1d
from pathlib import Path

from utils.data import Position, Rotation


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
        file_name = Path(__file__).parent / f"cache_xy_roadframe_iso8855_{self.img_params.width}x{self.img_params.height}_fov{self.img_params.fov_deg}.npy"
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
