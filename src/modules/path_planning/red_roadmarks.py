import numpy as np
import cv2

from datalink.data import Pose
from typing import Tuple, List
from scipy.interpolate import splprep, splev

# TODO: check for polyfit properly
import warnings

warnings.simplefilter("ignore", np.RankWarning)


# Camera
# -------------------------------------------------------------------------------------------------------------


def rpi_v2_intrinsic_matrix(image_shape: Tuple[int, int], binning_factor=2):
    assert binning_factor in [1, 2, 4], "Binning factor not in {1, 2, 4}"
    focal_len_mm = 3.04
    pixel_size_mm = 0.00112
    focal_len_pixels = focal_len_mm / pixel_size_mm
    # focal_len = 3.04 / 1e3
    # pixel_size = 0.00112 / 1e3
    # focal_len_pixels = focal_len / pixel_size

    # Binning combines nearby (2, 4..) pixel values into one
    fx = fy = focal_len_pixels
    cx, cy = image_shape[0] / 2, image_shape[1] / 2

    return np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    )


def rpi_v2_intrinsic_matrix_from_fov(image_shape: Tuple[int, int]):
    """
    Calculate the intrinsic matrix for Raspberry Pi Camera V2 using FOV values.

    Args:
        image_shape: (height, width) in pixels

    Returns:
        3x3 intrinsic camera matrix
    """

    # RPI Camera V2 FOV in degrees
    fov_h_deg = 62.2  # Horizontal FOV
    fov_v_deg = 48.8  # Vertical FOV

    # Convert FOV to radians
    fov_h = np.deg2rad(fov_h_deg)
    fov_v = np.deg2rad(fov_v_deg)

    width, height = image_shape

    # Calculate focal lengths using FOV
    fx = (width / 2) / np.tan(fov_h / 2)
    fy = (height / 2) / np.tan(fov_v / 2)

    # Principal point (typically center of image)
    cx = width / 2
    cy = height / 2

    # return np.array([
    #     [631.00614559,   0.,         399.13995178],
    #     [0.,         630.71927485, 298.79517474],
    #     [0.,           0.,           1.        ]
    # ])

    # Create the intrinsic matrix
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def unreal_engine_intrinsic_matrix(image_shape: Tuple[int, int], fov_deg: int):
    fov = np.deg2rad(fov_deg)
    a = (image_shape[0] / 2.0) / np.tan(fov / 2.0)  # Alpha
    cx, cy = image_shape[0] / 2, image_shape[1] / 2
    return np.array(
        [
            [a, 0, cx],
            [0, a, cy],
            [0, 0, 1],
        ]
    )


class Camera:
    def __init__(
        self,
        pose: Pose,
        image_shape: Tuple[int, int],
        intrinsic_matrix: np.ndarray,
        extrinsic_matrix: np.ndarray = None,
    ):
        self.pose = pose
        self.position = self.pose.position
        self.rotation = self.pose.rotation
        self.image_shape = image_shape

        self.M = intrinsic_matrix
        self.M_i = np.linalg.inv(self.M)

        self.distortion_coeffs = np.array([0.14946983, -0.36958066, -0.01219314, -0.00706742, 0.19695792])
        # Get optimal camera matrix
        h, w = image_shape
        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.M, self.distortion_coeffs, (w, h), 1, (w, h)
        )

        self.R_rc = self.rotation_matrix()
        self.R_cr = self.R_rc.T

        self.T_rc = np.array([self.position.x, self.position.y, self.position.z])
        self.T_cr = self.T_rc * -1.0

        self.H_cr = Camera.homogenous_transformation_matrix(R=self.R_cr, T=self.T_cr)
        self.H_rc = np.linalg.inv(self.H_cr)

        self.road_normal_camera_frame = self.R_rc @ np.array([0, 1, 0])

        xyz_roadframe = self.image2xyz_roadframe()
        self.image_xyz_roadframe = xyz_roadframe.reshape(*self.image_shape, 3)

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        """Undistort an image using the camera's distortion coefficients."""     
        # Undistort the image
        dst = cv2.undistort(img, self.M, self.distortion_coeffs, None, self.new_camera_matrix)
        return dst

    def update_camera(self):
        self.R_rc = self.rotation_matrix()
        self.R_cr = self.R_rc.T

        self.T_rc = np.array([self.position.x, self.position.y, self.position.z])
        self.T_cr = self.T_rc * -1.0

        self.H_cr = Camera.homogenous_transformation_matrix(R=self.R_cr, T=self.T_cr)
        self.H_rc = np.linalg.inv(self.H_cr)

        self.road_normal_camera_frame = self.R_rc @ np.array([0, 1, 0])

        xyz_roadframe = self.image2xyz_roadframe()
        self.image_xyz_roadframe = xyz_roadframe.reshape(*self.image_shape, 3)

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

    @staticmethod
    def homogenous_transformation_matrix(R, T):
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = T
        return H

    def uv2xyz_camframe(self, u: int, v: int) -> np.ndarray:
        image_point = [u, v, 1]
        direction_vector = self.M_i @ image_point
        scaling_factor = self.position.y / (self.road_normal_camera_frame @ direction_vector)
        xyz = scaling_factor * direction_vector
        return xyz

    def xyz_camframe2uv(self, xyz: np.ndarray) -> np.ndarray:
        uvw = self.M @ xyz
        uvw /= uvw[2]
        u, v = int(uvw[0]), int(uvw[1])
        return np.array([u, v])

    def uv2xyzw_roadframe(self, u: int, v: int) -> np.ndarray:
        xyz = self.uv2xyz_camframe(u, v)
        xyzw = Camera.cart2homo(xyz)
        return self.road2road_iso8855(self.H_cr @ xyzw)

    def road2road_iso8855(self, xyzw):
        return np.array([xyzw[2], -xyzw[0], -xyzw[1], xyzw[3]])

    def road_iso88552_road(self, xyzw):
        return np.array([-xyzw[1], -xyzw[2], xyzw[0], xyzw[3]])

    def xyzw_roadframe2uv(self, xyzw):
        xyzw_camframe = self.H_rc @ self.road_iso88552_road(xyzw)
        xyz_camframe = Camera.homo2cart(xyzw_camframe)
        return self.xyz_camframe2uv(xyz_camframe)

    def image2xyz_roadframe(self) -> np.ndarray:
        # 1. Create meshgrid of all pixel coordinates
        u_indices = np.arange(self.image_shape[0])
        v_indices = np.arange(self.image_shape[1])
        u_grid, v_grid = np.meshgrid(u_indices, v_indices, indexing="ij")

        # 2. Reshape to (N, 2) where N = width * height
        pixel_coords = np.stack([u_grid.flatten(), v_grid.flatten()], axis=-1)

        # 3. Add homogeneous coordinate and transform to camera ray directions
        pixel_coords_h = np.column_stack([pixel_coords, np.ones(pixel_coords.shape[0])])
        rays = pixel_coords_h @ self.M_i.T

        # 4. Compute scaling factors for all points
        # road_normal · ray = cos(angle) * |ray|
        denominators = np.sum(self.road_normal_camera_frame * rays, axis=1)
        scaling_factors = self.position.y / denominators

        # 5. Scale rays to get 3D points in camera frame
        xyzw_cam = rays * scaling_factors[:, np.newaxis]

        # 6. Convert to homogeneous coordinates
        xyzw_cam = np.column_stack([xyzw_cam, np.ones(xyzw_cam.shape[0])])

        # 7. Transform to road frame
        xyzw_road = xyzw_cam @ self.H_cr.T

        # 8. Convert to ISO8855 coordinate system (z, -x, -y, w)
        xyz_iso8855 = np.column_stack([xyzw_road[:, 2], -xyzw_road[:, 0], -xyzw_road[:, 1]])
        return xyz_iso8855

    @staticmethod
    def cart2homo(xyz: np.ndarray, w: float = 1) -> np.ndarray:
        return np.append(xyz, w)

    @staticmethod
    def homo2cart(xyzw: np.ndarray) -> np.ndarray:
        if np.isclose(xyzw[3], 0):
            print(f"[WARNING] Homogenous coordinates {xyzw} with w ~= 0")
            xyzw[:3]
        else:
            return (xyzw / xyzw[3])[:3]


# Planner
# -------------------------------------------------------------------------------------------------


class HSVColorFilter:
    """Detects given colors in images."""

    def __init__(
        self,
        color_ranges: List[Tuple[List]],
        morph_kernel=(2, 2),
        morph_open=True,
        morph_close=True,
    ):
        """
        Args:
            color_ranges: [(lower_bound, upper_bound), ...] where bound = [H, S, V]
            kernel_size: Size of kernel for morphological operations (default: (2, 2))
            morph_open: Apply morphological opening (default: True)
            morph_close: Apply morphological closing (default: True)
        """
        self.color_ranges = color_ranges
        self.morph_kernel = np.ones(morph_kernel, np.uint8)
        self.morph_open = morph_open
        self.morph_close = morph_close

    def apply(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(img_hsv, self.color_ranges[0][0], self.color_ranges[0][1])
        for lower, upper in self.color_ranges[1:]:
            mask = mask + cv2.inRange(img_hsv, lower, upper)

        if self.morph_open:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        if self.morph_close:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)

        img_hsv_filtered = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
        return cv2.cvtColor(img_hsv_filtered, cv2.COLOR_HSV2BGR)

    @classmethod
    def new_red(cls):
        # Red = Hue around 360 deg (179 - max value in OpenCV)
        color_ranges = [
            (np.array([0, 50, 50]), np.array([6, 255, 255])),
            (np.array([173, 50, 50]), np.array([180, 255, 255])),
        ]
        return cls(color_ranges)

    @classmethod
    def new_bright(cls):
        # Any Hue and Saturation, high Value
        # TODO: could be make simpler and more efficient converting to grayscale and using intensity
        color_ranges = [(np.array([0, 0, 203]), np.array([180, 255, 255]))]
        return cls(color_ranges)


class RoadmarksPlannerConfig:
    def __init__(self, roadmark_min_area: float, roadmark_max_count: int):
        self.roadmark_min_area = roadmark_min_area
        self.roadmark_max_count = roadmark_max_count


class RoadmarksPlanner:
    def __init__(self, camera: Camera, filter: HSVColorFilter, config: RoadmarksPlannerConfig):
        self.camera = camera
        self.filter = filter
        self.config = config
        # Intermediates
        self.img_filtered = None
        self.roadmarks_imgframe = None
        self.roadmarks_roadframe = None
        self.path_roadframe = None

    def set_config(self, config: RoadmarksPlannerConfig):
        self.config = config

    def update(self, img: np.ndarray):
        self.img_filtered = self.filter.apply(img)
        self.roadmarks_imgframe = self.get_roadmark_positions(self.img_filtered)
        self.roadmarks_roadframe = np.array(
            [self.camera.image_xyz_roadframe[u, v][:2] for u, v in self.roadmarks_imgframe]
        )
        self.path_roadframe = self.polyline_fit_path(self.roadmarks_roadframe)

    def get_roadmark_positions(self, bgr_filtered: np.ndarray) -> np.ndarray:
        # TODO: find better way to detect valid roadmarks
        img_gray = cv2.cvtColor(bgr_filtered, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roadmark_centers = []
        for c in contours:
            if len(roadmark_centers) == self.config.roadmark_max_count:
                break
            if cv2.contourArea(c) < self.config.roadmark_min_area:
                continue
            u, v, width, height = cv2.boundingRect(c)
            uc, vc = (u + width // 2, v + height // 2)
            roadmark_centers.append((uc, vc))
        return np.array(roadmark_centers)

    # def polyline_fit_path(self, roadmarks: np.ndarray) -> np.ndarray:
    #     coeffs = np.polyfit(x=roadmarks[:, 0], y=roadmarks[:, 1], deg=3)
    #     f = np.poly1d(coeffs)
    #     step = abs(roadmarks[-1][0] - roadmarks[0][0]) / 100
    #     xs = np.arange(roadmarks[0][0], roadmarks[-1][0], step)
    #     ys = f(xs)
    #     return np.column_stack((xs, ys))


    def polyline_fit_path(self, roadmarks: np.ndarray) -> np.ndarray:
        """
        Fits a smooth path through roadmarks using parametric cubic splines.
        Handles complex paths including loops and vertical segments.

        Args:
            roadmarks: Nx2 array of (x,y) roadmark coordinates

        Returns:
            Nx2 array of points forming a smooth path
        """

        if len(roadmarks) < 2:
            return roadmarks

        # Parametric spline representation (handles vertical segments and loops)
        tck, u = splprep([roadmarks[:, 0], roadmarks[:, 1]], s=0, k=min(3, len(roadmarks) - 1))

        num_points = 50
        u_new = np.linspace(0, 1, num_points)
        x_spline, y_spline = splev(u_new, tck)

        return np.column_stack((x_spline, y_spline))

    # def polyline_fit_path(self, roadmarks: np.ndarray) -> np.ndarray:
    #     """
    #     Fits a smooth path through roadmarks using a B-spline approximation.
    #     Handles complex paths including loops and vertical segments.

    #     Args:
    #         roadmarks: Nx2 array of (x,y) roadmark coordinates

    #     Returns:
    #         Nx2 array of points forming a smooth path
    #     """

    #     if len(roadmarks) < 2:
    #         return roadmarks
        
    #     print(roadmarks)

    #     # Fit a B-spline curve to the roadmarks
    #     num_points = 100  # Number of points in the resulting smooth path

    #     # Generate a uniform parameterization for the roadmarks
    #     t = np.linspace(0, 1, len(roadmarks))

    #     # Fit the B-spline
    #     x_spline = np.interp(np.linspace(0, 1, num_points), t, roadmarks[:, 0])
    #     y_spline = np.interp(np.linspace(0, 1, num_points), t, roadmarks[:, 1])

    #     return np.column_stack((x_spline, y_spline))