import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

warnings.simplefilter("ignore", np.RankWarning)


from modules.recorder import RecordReader
from datalink.data import Position, Rotation, Pose, ProcessedRealData
from utils.paths import record_path

from modules.path_planning.red_roadmarks import RoadmarksPlanner, RoadmarksPlannerConfig, Camera, rpi_v2_intrinsic_matrix, HSVColorFilter, unreal_engine_intrinsic_matrix

from threading import Thread
from enum import Enum
import tkinter as tk
from tkinter import ttk


# Foos
# -------------------------------------------------------------------------------------------------
# TODO: remove None checks - should be empty arrays for more clarity


def draw_debug_data(image, planner: RoadmarksPlanner, camera) -> np.ndarray:
    image_copy = image.copy()

    # Roadmark dots
    for u, v in planner.roadmarks_imgframe:
        cv2.circle(image_copy, (u, v), 5, (0, 255, 0), -1)

    path_imgframe = np.array([camera.xyzw_roadframe2uv(np.array([*xy, 0, 1]))[:2] for xy in planner.path_roadframe])

    # Planned path
    for i in range(len(path_imgframe) - 1):
        cv2.line(image_copy, path_imgframe[i], path_imgframe[i + 1], (0, 255, 0), 2)

    return image_copy


def plot_roadmarks(roadmarks: np.ndarray, path: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.scatter(roadmarks[:, 0], roadmarks[:, 1], color="red", label="Roadmarks")
    plt.plot(path[:, 0], path[:, 1], color="blue", label="Fitted Path")
    plt.xlabel("X (Road Frame)")
    plt.ylabel("Y (Road Frame)")
    plt.title("Roadmarks and Fitted Path")
    plt.legend()
    plt.grid()
    plt.show()


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

# Main
# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    reader = RecordReader()
    # path_roadframe = record_path("1743006114457116600")
    # path_roadframe = record_path("1743458155872706100")
    # path_roadframe = record_path("1743459421861391500")
    # path_roadframe = record_path("1743468528428846300")
    path_roadframe = record_path("1743006114457116600")
    data: list = reader.read_all(path_roadframe, ProcessedRealData)

    image_shape = (820, 616)
    camera = Camera(
        pose=Pose(position=Position(0, 0.125, 0), rotation=Rotation(0, -15.1, 0)),
        image_shape=image_shape,
        intrinsic_matrix=rpi_v2_intrinsic_matrix(image_shape=image_shape),
    )

    config = RoadmarksPlannerConfig(roadmark_min_area=50, roadmark_max_count=3)
    planner = RoadmarksPlanner(camera=camera, filter=HSVColorFilter.new_bright(), config=config)

    dt = 1/30
    while True:
        for frame in data:

            jpg = frame.original.sensor_fusion.camera.jpg
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

            planner.update(img)

            plot_roadmarks(planner.roadmarks_roadframe, planner.path_roadframe)

            updated_image = draw_debug_data(img, planner=planner, camera=camera)
            stacked_image = np.hstack((img, updated_image, planner.img_filtered))
            cv2.imshow("Original | Debug | Filtered", stacked_image)
            cv2.waitKey(int(dt*1e3))

    cv2.destroyAllWindows()

