import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

warnings.simplefilter("ignore", np.RankWarning)


from modules.recorder import RecordReader
from datalink.data import Position, Rotation, Pose, ProcessedRealData
from utils.paths import record_path

from src.modules.path_planning.roadmarks import (
    RoadmarksPlannerConfig,
    Camera,
    rpi_v2_intrinsic_matrix,
    HSVColorFilter,
)
from scipy.interpolate import splprep, splev


# Foos
# -------------------------------------------------------------------------------------------------
# TODO: remove None checks - should be empty arrays for more clarity


def plot_roadmarks(roadmarks: np.ndarray, path: np.ndarray):
    """Convert matplotlib plot to OpenCV image without blocking"""

    # Create figure with specific size
    plt.figure(figsize=(8, 6), dpi=100)

    # Create the plot
    if len(roadmarks) and len(path):
        plt.scatter(roadmarks[:, 0], roadmarks[:, 1], color="red", label="Roadmarks")
        plt.plot(path[:, 0], path[:, 1], color="blue", label="Fitted Path")
    plt.xlabel("X (Road Frame)")
    plt.ylabel("Y (Road Frame)")
    plt.title("Roadmarks and Fitted Path")
    plt.legend()
    plt.grid()
    # Note: The function signature should be updated to accept target_shape:
    # def plot_roadmarks(roadmarks: np.ndarray, path: np.ndarray, target_shape: tuple):

    # Get target shape (e.g., camera.image_shape) passed as 'target_shape' argument
    target_height, target_width = 616, 820
    # Get the current figure and its DPI
    fig = plt.gcf()
    dpi = fig.get_dpi()
    # Calculate the required figure size in inches to match the target pixel dimensions
    fig_width_inches = target_width / dpi
    fig_height_inches = target_height / dpi
    # Set the figure size before rendering
    fig.set_size_inches(fig_width_inches, fig_height_inches)
    # Convert plot to image
    fig = plt.gcf()
    fig.canvas.draw()

    # Convert canvas to image using buffer_rgba() which returns an RGBA buffer
    rgba_buf = np.array(fig.canvas.buffer_rgba())

    # Convert RGBA to BGR for OpenCV
    plot_img = cv2.cvtColor(rgba_buf, cv2.COLOR_RGBA2BGR)

    # Close the figure to prevent memory leaks
    plt.close(fig)

    return plot_img


# def visualize_projection_steps(camera: Camera, original_image=None):
#     """
#     Visualizes the steps of the image2xy_roadframe_iso8855_fast transformation
#     with informative plots for each major stage.

#     Args:
#         camera: Camera object
#         original_image: Optional RGB image to show in the visualization
#     """
#     fig = plt.figure(figsize=(20, 15))

#     # Get intermediate results from transformation steps
#     # 1. Create meshgrid of pixel coordinates
#     u_indices = np.arange(camera.image_shape[0])
#     v_indices = np.arange(camera.image_shape[1])
#     u_grid, v_grid = np.meshgrid(u_indices, v_indices, indexing="ij")

#     # For visualization, sample points to avoid overcrowding
#     sample_rate = 30
#     u_sampled = u_grid[::sample_rate, ::sample_rate]
#     v_sampled = v_grid[::sample_rate, ::sample_rate]

#     # 2. Get rays
#     pixel_coords = np.stack([u_sampled.flatten(), v_sampled.flatten()], axis=-1)
#     pixel_coords_h = np.column_stack([pixel_coords, np.ones(pixel_coords.shape[0])])
#     rays = pixel_coords_h @ camera.M_i.T

#     # 3. Get 3D points in camera frame
#     denominators = np.sum(camera.road_normal_camera_frame * rays, axis=1)
#     scaling_factors = camera.position.y / denominators
#     xyz_cam = rays * scaling_factors[:, np.newaxis]

#     # 4. Convert to road frame
#     xyzw_cam = np.column_stack([xyz_cam, np.ones(xyz_cam.shape[0])])
#     xyzw_road = xyzw_cam @ camera.H_cr.T

#     # 5. Convert to ISO8855
#     xy_iso8855 = np.column_stack([xyzw_road[:, 2], -xyzw_road[:, 0]])

#     # PLOT 1: Original image with sampled grid points
#     ax1 = fig.add_subplot(2, 2, 1)
#     if original_image is not None:
#         ax1.imshow(original_image)
#     ax1.scatter(u_sampled.flatten(), v_sampled.flatten(), c="r", s=2)
#     ax1.set_title("1. Image Grid Points")
#     ax1.set_xlabel("u (pixels)")
#     ax1.set_ylabel("v (pixels)")

#     # PLOT 2: Camera rays visualization (3D)
#     ax2 = fig.add_subplot(2, 2, 2, projection="3d")
#     ray_scale = 0.1  # Scale factor for ray visualization

#     # Plot origin (camera)
#     ax2.scatter([0], [0], [0], c="r", s=100, label="Camera")

#     # Plot rays
#     for i in range(0, len(rays), 10):  # Plot every 10th ray to avoid clutter
#         ray = rays[i]
#         ax2.plot(
#             [0, ray_scale * ray[0]],
#             [0, ray_scale * ray[1]],
#             [0, ray_scale * ray[2]],
#             "b-",
#             alpha=0.3,
#         )

#     # Plot road plane
#     x_grid, z_grid = np.meshgrid(np.linspace(-0.5, 0.5, 10), np.linspace(-0.5, 0.5, 10))
#     y_grid = np.zeros_like(x_grid)  # Road is at y=0 in camera frame
#     ax2.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color="g")

#     ax2.set_title("2. Camera Rays")
#     ax2.set_xlabel("X (Camera Frame)")
#     ax2.set_ylabel("Y (Camera Frame)")
#     ax2.set_zlabel("Z (Camera Frame)")
#     ax2.set_xlim([-0.5, 0.5])
#     ax2.set_ylim([-0.5, 0.5])
#     ax2.set_zlim([-0.5, 0.5])

#     # PLOT 3: Points in camera frame
#     ax3 = fig.add_subplot(2, 2, 3, projection="3d")
#     ax3.scatter(xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2], c="b", s=2)
#     ax3.set_title("3. Intersection Points (Camera Frame)")
#     ax3.set_xlabel("X (Camera Frame)")
#     ax3.set_ylabel("Y (Camera Frame)")
#     ax3.set_zlabel("Z (Camera Frame)")

#     # PLOT 4: Points in road frame (ISO8855)
#     ax4 = fig.add_subplot(2, 2, 4)
#     scatter = ax4.scatter(
#         xy_iso8855[:, 0], xy_iso8855[:, 1], c=v_sampled.flatten(), s=10, cmap="rainbow"
#     )
#     ax4.set_title("4. Final Road Coordinates (ISO8855)")
#     ax4.set_xlabel("X (Road Frame)")
#     ax4.set_ylabel("Y (Road Frame)")
#     ax4.grid(True)
#     plt.colorbar(scatter, ax=ax4, label="v coordinate (row)")

#     plt.tight_layout()
#     return fig


# def test_image2roadframe():
#     start_time = time.time()
#     xy_roadframe_iso8855 = camera.image2xy_roadframe_iso8855()
#     end_time = time.time()
#     print("Time taken by image2xy_roadframe_iso8855:", end_time - start_time, "seconds")

#     # Measure time for image2xy_roadframe_iso8855_fast
#     start_time = time.time()
#     xy_roadframe_iso8855_fast = camera.image2xy_roadframe_iso8855_fast()
#     end_time = time.time()
#     print("Time taken by image2xy_roadframe_iso8855_fast:", end_time - start_time, "seconds")

#     # Compare the results
#     difference = np.abs(xy_roadframe_iso8855 - xy_roadframe_iso8855_fast)
#     max_difference = np.max(difference)

#     print(
#         "Max difference between image2xy_roadframe_iso8855 and image2xy_roadframe_iso8855_fast:",
#         max_difference,
#     )

#     if max_difference < 1e-6:
#         print("The results are consistent!")
#     else:
#         print("The results differ. Investigate further.")


def draw_debug_data(image, planner: "RoadmarksPlanner", camera) -> np.ndarray:
    image_copy = image.copy()

    # Roadmark dots
    for u, v in planner.roadmarks_imgframe:
        cv2.circle(image_copy, (u, v), 5, (0, 255, 0), -1)

    path_imgframe = np.array(
        [camera.xyzw_roadframe2uv(np.array([*xy, 0, 1]))[:2] for xy in planner.path_roadframe]
    )

    # Planned path
    for i in range(len(path_imgframe) - 1):
        cv2.line(image_copy, path_imgframe[i], path_imgframe[i + 1], (0, 255, 0), 2)

    return image_copy


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
        # Apply outlier filtering
        self.roadmarks_roadframe = self.filter_outliers(self.roadmarks_roadframe)

        self.path_roadframe = self.fit_path_param_cubic_spline(self.roadmarks_roadframe)

    def filter_outliers(self, roadmarks: np.ndarray) -> np.ndarray:
        """
        Filters out outlier roadmarks using a simple distance-based approach.

        Args:
            roadmarks: Nx2 array of roadmark coordinates in road frame

        Returns:
            Filtered array of roadmark coordinates
        """
        # Filter by max distance from origin (0,0) in road frame
        distances_from_origin = np.linalg.norm(roadmarks, axis=1)
        print(f"{roadmarks}\n{distances_from_origin}")
        roadmarks = roadmarks[distances_from_origin <= self.config.roadmark_max_distance]

        if len(roadmarks) < 3:  # Need at least 3 points for meaningful filtering
            return roadmarks

        # Calculate distances between consecutive pairs of points
        distances = []
        for i in range(len(roadmarks) - 1):
            dist = np.linalg.norm(roadmarks[i + 1] - roadmarks[i])
            distances.append(dist)

        print(distances)

        # Find median distance (more robust than mean)
        median_distance = np.median(distances)
        if len(distances) == 2:
            median_distance = min(distances)

        # Filter out points that are too far from their neighbors
        filtered_roadmarks = []
        outlier_flags = [False] * len(roadmarks)
        for i in range(len(roadmarks)):

            # Check distance to previous non-outlier point
            if i > 0:
                j = i - 1
                while outlier_flags[j] and j > 0:
                    j -= 1
                prev_dist = np.linalg.norm(roadmarks[i] - roadmarks[j])
                if prev_dist > 3.0 * median_distance:  # Threshold: 3x median
                    print(f"i: {i} prev: {prev_dist} point: {roadmarks[i]}")
                    outlier_flags[i] = True

            if not outlier_flags[i]:
                filtered_roadmarks.append(roadmarks[i])

        print(roadmarks)
        print(filtered_roadmarks)
        return np.array(filtered_roadmarks)

    def get_roadmark_positions(self, bgr_filtered: np.ndarray) -> np.ndarray:
        # TODO: find better way to detect valid roadmarks
        img_gray = cv2.cvtColor(bgr_filtered, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roadmark_centers = []
        for c in contours:
            if len(roadmark_centers) == self.config.roadmark_max_count:
                break
            c_area = cv2.contourArea(c)
            if c_area < self.config.roadmark_min_area:
                continue
            if c_area > self.config.roadmark_max_area:
                continue
            u, v, width, height = cv2.boundingRect(c)
            uc, vc = (u + width // 2, v + height // 2)
            roadmark_centers.append((uc, vc))
        return np.array(roadmark_centers)

    def fit_path_simple_polyline(self, roadmarks: np.ndarray) -> np.ndarray:
        coeffs = np.polyfit(x=roadmarks[:, 0], y=roadmarks[:, 1], deg=3)
        f = np.poly1d(coeffs)
        step = abs(roadmarks[-1][0] - roadmarks[0][0]) / 100
        xs = np.arange(roadmarks[0][0], roadmarks[-1][0], step)
        ys = f(xs)
        return np.column_stack((xs, ys))

    def fit_path_param_cubic_spline(self, roadmarks: np.ndarray) -> np.ndarray:
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

        tck, u = splprep([roadmarks[:, 0], roadmarks[:, 1]], s=0, k=min(3, len(roadmarks) - 1))
        num_points = 50
        u_new = np.linspace(0, 1, num_points)
        x_spline, y_spline = splev(u_new, tck)

        return np.column_stack((x_spline, y_spline))


# Main
# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    reader = RecordReader()
    path_roadframe = record_path("1744726034609610300")
    data: list = reader.read_all(path_roadframe, ProcessedRealData)

    image_shape = (820, 616)
    camera = Camera(
        pose=Pose(position=Position(0, 0.135, 0), rotation=Rotation(0, 0.01, 0)),
        image_shape=image_shape,
        intrinsic_matrix=rpi_v2_intrinsic_matrix(image_shape=image_shape),
    )

    config = RoadmarksPlannerConfig(
        roadmark_min_area=50, roadmark_max_distance=10, roadmark_max_area=10000, roadmark_max_count=6
    )
    planner = RoadmarksPlanner(camera=camera, filter=HSVColorFilter.new_bright(), config=config)

    dt = 1 / 30
    frame_index = 0
    paused = False

    print("Controls:")
    print("  Space: Pause/Resume")
    print("  Esc: Exit")

    while True:
        if frame_index >= len(data):
            frame_index = 0  # Loop back to beginning

        # Get current frame
        frame = data[frame_index]
        jpg = frame.original.sensor_fusion.camera.jpg
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

        # Process frame
        planner.update(img)

        # Get the roadmarks plot as an OpenCV image
        plot_img = plot_roadmarks(planner.roadmarks_roadframe, planner.path_roadframe)

        # Create debug view
        updated_image = draw_debug_data(img, planner=planner, camera=camera)

        # Create display with plot integrated
        top_row = np.hstack((img, updated_image))
        bottom_row = np.hstack((planner.img_filtered, plot_img))

        # Resize bottom row to match top row width
        if top_row.shape[1] != bottom_row.shape[1]:
            scale = top_row.shape[1] / bottom_row.shape[1]
            bottom_row = cv2.resize(
                bottom_row, (top_row.shape[1], int(bottom_row.shape[0] * scale))
            )

        # Stack rows vertically
        full_display = np.vstack((top_row, bottom_row))

        # Show image with playback status
        status = "PAUSED" if paused else "Playing"
        cv2.imshow("window", full_display)

        print(frame_index)

        # Handle keyboard input (1ms wait for key)
        key = cv2.waitKey(1) & 0xFF

        # Space = toggle pause
        if key == 32:  # Space
            paused = not paused
            print("Playback", "paused" if paused else "resumed")
            while True:
                if cv2.waitKey(1) & 0xFF == 32:
                    break

        # Esc = exit
        elif key == 27:  # Escape
            print("Exiting...")
            break

        # Advance frame if not paused
        if not paused:
            frame_index += 1
            time.sleep(dt)  # Control playback speed

    cv2.destroyAllWindows()
