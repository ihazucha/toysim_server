from time import time_ns
import cv2
import numpy as np
import matplotlib.pyplot as plt

from planner import (
    get_last_record,
    red_mask_thresh,
    write_last_record_first_frame,
    read_last_record_first_frame,
)
from utils.data import Position, Rotation, SimData
from scipy.interpolate import interp1d
from control import PurePursuitPlusPID

CAM_FOV = 90
CAM_TILT_ANGLE = -15
CAM_HEIGHT = 250
IMG_WIDTH = 640
IMG_HEIGHT = 480


def find_red_landmarks(filtered_bgr):
    gray = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    MIN_AREA = 10
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    landmarks = [cv2.boundingRect(contour) for contour in contours]
    centers = [(x + w // 2, y + h // 2) for x, y, w, h in landmarks]
    centers = [c for c in centers if (c[0] > (IMG_WIDTH * 0.25)) and (c[0] < (IMG_WIDTH * 0.75))]
    return centers[:5] if len(centers) > 5 else centers


def draw_contour_centers(img, contour_centers):
    for cc in contour_centers:
        cv2.circle(img, cc, 5, (0, 255, 0), -1)
        cv2.putText(
            img,
            f"({cc[0]}, {cc[1]})",
            (cc[0] + 10, cc[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


# In Unreal Engine, the default coordinate system goes:
# (X, Y, Z) = (forward, right, up)
# Reference frames: World, Vehicle, Camera, Road


class Camera:
    def __init__(
        self,
        position: Position,
        rotation: Rotation,
        img_width: int = 640,
        img_height: int = 480,
        fov_deg: float = 90,
    ):
        self.position = position
        self.rotation = rotation
        self.img_width = img_width
        self.img_height = img_height
        self.fov_deg = fov_deg

        self.M = self.intrinsic_matrix()
        self.M_i = np.linalg.inv(self.M)

        roll = np.deg2rad(self.rotation.roll)
        pitch = np.deg2rad(self.rotation.pitch)
        yaw = np.deg2rad(self.rotation.yaw)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        self.R_rc = np.array(
            [
                [cr * cy + sp * sr + sy, cr * sp * sy - cy * sr, -cp * sy],
                [cp * sr, cp * cr, sp],
                [cr * sy - cy * sp * sr, -cr * cy * sp - sr * sy, cp * cy],
            ]
        )
        self.R_cr = self.R_rc.T
        self.T_cr = np.array([0, -self.position.y, 0])  # y = camera height
        self.H_cr = np.eye(4)
        self.H_cr[:3, :3] = self.R_cr
        self.H_cr[:3, 3] = self.T_cr
        self.H_rc = np.linalg.inv(self.H_cr)
        self.road_normal_camera_frame = self.R_rc @ np.array([0, 1, 0])

    def intrinsic_matrix(self) -> np.ndarray:
        fov_rad = np.deg2rad(self.fov_deg)
        alpha = (self.img_width / 2.0) / np.tan(fov_rad / 2.0)
        return np.array(
            [[alpha, 0, self.img_width / 2], [0, alpha, self.img_height / 2], [0, 0, 1]]
        )

    def cam2road(self, vec: np.ndarray) -> np.ndarray:
        rotated = self.R_cr @ vec
        translated = rotated + self.T_cr
        return translated

    def road2cam(self, vec: np.ndarray) -> np.ndarray:
        untranslated = vec - self.T_cr
        unrotated = self.R_rc @ untranslated
        return unrotated

    def uv2xyz_camframe(self, u: int, v: int) -> np.ndarray:
        image_point = [u, v, 1]
        direction_vector = self.M_i @ image_point
        scaling_factor = self.position.y / (self.road_normal_camera_frame @ direction_vector)
        xyz_c = scaling_factor * direction_vector
        return xyz_c

    def xyz_camframe2uv(self, xyz: np.ndarray) -> np.ndarray:
        uvw = self.M @ xyz
        u = uvw[0] / uvw[2]
        v = uvw[1] / uvw[2]
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

    def image_xy(self) -> np.ndarray:
        xy = []
        for v in range(self.img_height):
            for u in range(self.img_width):
                X, Y, _ = self.uv2xyz_roadframe(u, v)
                xy.append([X, Y])
        xy = np.array(xy)
        return xy

    def v_threshold_by_distance(self, distance: float):
        road_point_road_frame = np.array([0, 0, distance, 1])
        road_point_cam_frame = self.H_rc @ road_point_road_frame
        uv_vec = self.M @ road_point_cam_frame[:3]
        uv_vec /= uv_vec[2]
        cut_v = uv_vec[1]
        return cut_v

    def canera_to_world(self, vec_vehicle_frame: np.ndarray):
        return self.H_cr @ vec_vehicle_frame

    def world_to_camera(self, vec_world_frame: np.ndarray):
        return self.H_rc @ vec_world_frame


def plot_landmarks_iso8855(xyzs: np.ndarray) -> None:
    plt.scatter(xyzs[:, 0], xyzs[:, 1], c="blue", marker="x")
    plt.xlabel("X (long) [cm]")
    plt.ylabel("Y (lat) [cm]")
    plt.title("Landmarks in Road Frame")
    plt.grid(True)
    plt.show()


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


def pixel_target_tool(cam, bgr, transformed_depth, depth):
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
    cv2.createTrackbar("U", "depth", pixel_target[0], IMG_WIDTH - 1, update_pixel_target)
    cv2.createTrackbar("V", "depth", pixel_target[1], IMG_HEIGHT - 1, update_pixel_target)
    cv2.imshow("depth", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class RedRoadmarksPathPlanner:
    def __init__(self): ...

    def step(rgb_image: np.ndarray) -> np.ndarray:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        bgr_filtered = RedRoadmarksPathPlanner.red_color_thresh_filter(bgr_image)

        landmarks = find_red_landmarks(bgr_filtered)

    @staticmethod
    def find_roadmarks(
        bgr_filtered: np.ndarray, min_area: float = 10.0, max_roadmarks: int = 5
    ) -> list:
        gray = cv2.cvtColor(bgr_filtered, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roadmark_centers = []
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            xc, yc = (x + w // 2, y + h // 2)
            if (xc < (IMG_WIDTH * 0.25)) or (xc > (IMG_WIDTH * 0.75)):
                continue
            roadmark_centers.append((xc, yc))

        return roadmark_centers[:max_roadmarks] if max_roadmarks > 0 else roadmark_centers

    @staticmethod
    def red_color_thresh_filter(bgr_image):
        # Red in HSV space is depicted by Hue around 360 deg
        # which is around 179 (max value in OpenCV)
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 50, 50])
        upper_red = np.array([7, 255, 255])
        mask0 = cv2.inRange(hsv_image, lower_red, upper_red)

        lower_red = np.array([172, 50, 50])
        upper_red = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red, upper_red)

        mask = mask0 + mask1
        hsv_filtered = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
        return cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR)


def main():
    # write_last_record_first_frame()
    cam = Camera(Position(0, 251, 0), Rotation(0, -14.33, 0))
    records: list[SimData] = get_last_record()

    controller = PurePursuitPlusPID()

    steering_setpoints = []
    speed_setpoints = []

    # print(records)
    for data in records:
        tstart = time_ns()
        depth = data.camera_data.depth_image
        transformed_depth = transform_depth_to_camera_center(depth, cam)
        tend = time_ns()
        dt = (tend - tstart) / 1e9
        # print(dt)

        rgb = data.camera_data.rgb_image
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        filtered_hsv = red_mask_thresh(hsv)
        filtered_bgr = cv2.cvtColor(filtered_hsv, cv2.COLOR_HSV2BGR)

        landmarks = find_red_landmarks(filtered_bgr)

        xyzs_road = []
        for lm in landmarks:
            xyz_cam = cam.uv2xyz_camframe(lm[0], lm[1])
            xyz_road = cam.uv2xyz_roadframe_iso8855(lm[0], lm[1])
            xyzs_road.append(xyz_road)
            dist = np.linalg.norm(xyz_cam)

            print(
                f"lm(x, y, d_gt, d, xyz_cam, xyz_road): ({lm[0]}, {lm[1]}, {transformed_depth[lm[1], lm[0]]}, {dist:.2f}, {[int(x) for x in xyz_cam]}, {[int(x) for x in xyz_road]})"
            )

        xyzs_road = np.array(xyzs_road)
        # Interpolate between the points to add more points to xyzs_road
        if len(xyzs_road) > 1:
            f_interp = interp1d(xyzs_road[:, 0], xyzs_road[:, 1], kind="linear")
            x_new = np.linspace(xyzs_road[0, 0], xyzs_road[-1, 0], num=10)
            y_new = f_interp(x_new)
            xyzs_road = np.column_stack((x_new, y_new))

            weights = np.array(range(len(xyzs_road)))
            poly_coeff = np.polyfit(x=xyzs_road[:, 0], y=xyzs_road[:, 1], deg=2, w=weights)
            polyline = np.poly1d(poly_coeff)
            polyline_xs = np.arange(xyzs_road[0][0], xyzs_road[-1][0], 50)
            polyline_ys = polyline(polyline_xs)
            # plt.plot(polyline_xs, polyline_ys, c="blue")
            # plt.scatter(xyzs_road[:, 0], xyzs_road[:, 1], marker="x")
            # plt.xlabel("X (long) [cm]")
            # plt.ylabel("Y (lat) [cm]")
            # plt.title("Landmark polyline in Road Frame")
            # plt.grid(True)
            # plt.show()

            polypoint_uvs = []
            for polypoint in zip(polyline_xs, polyline_ys):
                xyz = [*polypoint, 0]
                uv = cam.xyz_roadframe_iso88552uv(xyz)
                polypoint_uvs.append((int(uv[0]), int(uv[1])))
            # plot_landmarks_iso8855(xyzs_road)

            speed_mps = data.vehicle_data.speed / 1000
            steering = data.vehicle_data.steering_angle
            waypoints_m = np.column_stack((polyline_xs, polyline_ys)) / 1000
            setpoints = controller.get_control(
                waypoints=waypoints_m, speed=speed_mps, desired_speed=3, dt=0.3333
            )
            set_speed, set_steering_angle = setpoints[0], np.rad2deg(setpoints[1])
            speed_setpoints.append(set_speed)
            steering_setpoints.append(set_steering_angle)
            # print(f"\r(v, v_s) = ({speed_mps:5.1f}, {set_speed:5.1f}), (steer, _s) = ({steering:5.1f}, {set_steering_angle:5.1f})", end="")

            # Drawing
            # -------------------------------------------
            for i in range(len(polypoint_uvs) - 1):
                cv2.line(filtered_bgr, polypoint_uvs[i], polypoint_uvs[i + 1], (255, 0, 0), 2)

            draw_contour_centers(filtered_bgr, landmarks)
            combined_image = np.hstack((bgr, filtered_bgr))
            cv2.imshow("Combined Image", combined_image)
            if cv2.waitKey(33) & 0xFF == ord("q"):
                exit()
    # Plotting speed and steering setpoints
    plt.figure(figsize=(12, 6))

    # Speed setpoints plot
    plt.subplot(1, 2, 1)
    plt.plot(speed_setpoints, label="Speed Setpoints")
    plt.xlabel("Time Step")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed Setpoints Over Time")
    plt.legend()
    plt.grid(True)

    # Steering setpoints plot
    plt.subplot(1, 2, 2)
    plt.plot(steering_setpoints, label="Steering Setpoints", color="orange")
    plt.xlabel("Time Step")
    plt.ylabel("Steering Angle (degrees)")
    plt.title("Steering Setpoints Over Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
