import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from planner import red_mask_thresh

CAM_FOV = 90
CAM_TILT_ANGLE = -15
CAM_HEIGHT = 250 # cm
IMG_WIDTH = 640
IMG_HEIGHT = 480

def write():
    from utils.data import SimData, last_record_path
    from modules.recorder import RecordReader

    record_path = last_record_path()
    if record_path is None:
        print("[Planner] No records found")
        exit()
    data: list[SimData] = RecordReader.read(record_path=record_path)
    
    with open(os.path.join(os.path.dirname(__file__), "../data/sandbox/bgr_image.bin"), "wb") as f:
        bgr = cv2.cvtColor(data[0].camera_data.rgb_image, cv2.COLOR_RGB2BGR)
        f.write(bgr.tobytes())

    with open(os.path.join(os.path.dirname(__file__), "../data/sandbox/depth_image.bin"), "wb") as f:
        f.write(data[0].camera_data.depth_image.tobytes())


def read_bgr():
    with open(os.path.join(os.path.dirname(__file__), "../data/sandbox/bgr_image.bin"), "rb") as f:
        bgr = np.frombuffer(f.read(), dtype=np.uint8).reshape((480, 640, 3))
        return bgr

def read_depth():
    with open(os.path.join(os.path.dirname(__file__), "../data/sandbox/depth_image.bin"), "rb") as f:
        depth = np.frombuffer(f.read(), dtype=np.float16).reshape((480, 640))
        return depth

def find_red_landmarks(filtered_bgr):
    gray = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    MIN_CONTOUR_AREA = 10
    contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_CONTOUR_AREA]
    landmarks = [cv2.boundingRect(contour) for contour in contours]
    return [(x + w // 2, y + h // 2) for x, y, w, h in landmarks]

def plot_landmarks(landmarks):
    x_coords, y_coords = zip(*landmarks)
    plt.scatter(x_coords, y_coords, c='red', marker='o')
    plt.xlabel('X ground (cm)')
    plt.ylabel('Y ground (cm)')
    plt.title('Red Landmarks on Ground Plane')
    plt.grid(True)
    plt.show()

def draw_contour_centers(img, contour_centers):
    for cc in contour_centers:
        cv2.circle(img, cc, 5, (0, 255, 0), -1)
        cv2.putText(img, f"({cc[0]}, {cc[1]})", (cc[0] + 10, cc[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def pixel_to_angle(pixel, img_dim, fov):
    return (pixel - img_dim / 2) * (fov / img_dim)

def calculate_distance_from_camera(lm, cam_height, cam_tilt_angle, cam_fov, img_width, img_height):
    x_pixel, y_pixel = lm
    x_angle = pixel_to_angle(x_pixel, img_width, cam_fov)
    y_angle = pixel_to_angle(y_pixel, img_height, cam_fov * (img_height / img_width))

    y_angle_corrected = y_angle - cam_tilt_angle

    # Calculate the distance on the ground plane
    ground_distance = cam_height / np.tan(np.radians(y_angle_corrected))

    # Calculate the actual 3D distance using both x and y angles
    distance = ground_distance / np.cos(np.radians(x_angle))
    return distance

if __name__ == "__main__":
    # write()
    bgr = read_bgr()
    depth = read_depth()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    filtered_hsv = red_mask_thresh(hsv)
    filtered_bgr = cv2.cvtColor(filtered_hsv, cv2.COLOR_HSV2BGR)

    landmarks = find_red_landmarks(filtered_bgr)

    for lm in landmarks:
        distance = calculate_distance_from_camera(lm, CAM_HEIGHT, CAM_TILT_ANGLE, CAM_FOV, IMG_WIDTH, IMG_HEIGHT)
        print(f"lm(x, y, dist_real, dist_calc): ({lm[0]}, {lm[1]}, {depth[lm[1], lm[0]]}, {distance:.2f})")

    # plot_landmarks(landmarks)
    draw_contour_centers(filtered_bgr, landmarks)
    combined_image = np.hstack((bgr, filtered_bgr))
    cv2.imshow("Combined Image", combined_image)
    if cv2.waitKey(100000) & 0xFF == ord("q"):
        exit()