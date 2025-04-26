import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

warnings.simplefilter("ignore", np.RankWarning)


from modules.recorder import RecordReader
from datalink.data import Position, Rotation, Pose, ProcessedRealData
from utils.paths import record_path

from src.modules.path_planning.roadmarks import RoadmarksPlanner, RoadmarksPlannerConfig, Camera, rpi_v2_intrinsic_matrix, HSVColorFilter, unreal_engine_intrinsic_matrix
import glob
import os

def run_frames():
    reader = RecordReader()
    path_roadframe = record_path("1743005788249080400")
    data: list = reader.read_all(path_roadframe, ProcessedRealData)

    dt = 1/30
    counter = 0
    for frame in data:
        jpg = frame.original.sensor_fusion.camera.jpg
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

        cv2.imshow("Original | Debug | Filtered", img)
        counter += 1

        if cv2.waitKey(int(dt*1e3)) & 0xFF == ord('c'):
            save_path = f"camera_calibration/{counter}.jpg"
            cv2.imwrite(save_path, img)
            print(f"Saved: {save_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    def estimate_extrinsics():
        # Termination criteria for corner sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (6,5,0)
        objp = np.zeros((6*7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all images
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane

        images = glob.glob("camera_calibration/*.jpg")

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
                cv2.imshow("Chessboard", img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            print("Camera matrix:\n", mtx)
            print("Distortion coefficients:\n", dist)
            print("Rotation vectors:\n", rvecs)
            print("Translation vectors:\n", tvecs)
        else:
            print("Calibration failed.")

    estimate_extrinsics()