from utils.data import SimData, last_record_path
from modules.recorder import RecordReader

import cv2
import numpy as np
from time import sleep, time_ns

T = 1.0 / 30.0


def get_data() -> list[SimData]:
    record_path = last_record_path()
    if record_path is None:
        print("[Planner] No records found")
        exit()
    data: list[SimData] = RecordReader.read(record_path=record_path)
    return data


def threshold1_numpy(img):
    # Create a mask where red is dominant
    red_dominant_mask = (img[:, :, 0] > img[:, :, 1]) | (img[:, :, 0] > img[:, :, 2])
    # Apply the mask to the image
    img[~red_dominant_mask] = [0, 0, 0]
    return img


def hsv_thresh_sliders():
    cv2.createTrackbar("HMin", "image", 0, 179, lambda: ...)  # 0-179 for Opencv
    cv2.createTrackbar("HMax", "image", 0, 179, lambda: ...)
    cv2.createTrackbar("SMin", "image", 0, 255, lambda: ...)
    cv2.createTrackbar("SMax", "image", 0, 255, lambda: ...)
    cv2.createTrackbar("VMin", "image", 0, 255, lambda: ...)
    cv2.createTrackbar("VMax", "image", 0, 255, lambda: ...)

    cv2.setTrackbarPos("HMax", "image", 179)
    cv2.setTrackbarPos("SMax", "image", 255)
    cv2.setTrackbarPos("VMax", "image", 255)


# HMax set to 12
def hsv_thresh(hsv):
    hMin = cv2.getTrackbarPos("HMin", "image")
    sMin = cv2.getTrackbarPos("SMin", "image")
    vMin = cv2.getTrackbarPos("VMin", "image")

    hMax = cv2.getTrackbarPos("HMax", "image")
    sMax = cv2.getTrackbarPos("SMax", "image")
    vMax = cv2.getTrackbarPos("VMax", "image")

    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(bgr, bgr, mask=mask)
    return output


def red_mask_thresh(hsv):
    # Red in HSV space is depicted by Hue around 360 deg
    # which is around 179 (max value in OpenCV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([7, 255, 255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([172, 50, 50])
    upper_red = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask0 + mask1
    result = cv2.bitwise_and(hsv, hsv, mask=mask)

    return result


def get_marker_contours(red_bgr):
    # Convert the filtered image to grayscale
    gray = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2GRAY)
    # Find contours in the grayscale image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    MIN_CONTOUR_AREA = 10
    contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_CONTOUR_AREA]
    return contours


def get_contour_centers(contours):
    contour_centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            contour_centers.append((cX, cY))
    return contour_centers


def draw_poly_curve(img, contour_centers):
    if len(contour_centers) > 1:
        # Fit a polynomial through the contour centers
        contour_centers_np = np.array(contour_centers, dtype=np.float32)
        x = contour_centers_np[:, 0]
        y = contour_centers_np[:, 1]
        poly_coeff = np.polyfit(x, y, 2)  # Fit a 2nd degree polynomial
        poly = np.poly1d(poly_coeff)

        # Generate points along the polynomial curve
        x_new = np.linspace(x.min(), x.max(), 100)
        y_new = poly(x_new)

        # Draw the polynomial curve
        for i in range(len(x_new) - 2):
            cv2.line(
                img,
                (int(x_new[i]), int(y_new[i])),
                (int(x_new[i + 1]), int(y_new[i + 1])),
                (0, 255, 0),
                2,
            )


def draw_contour_centers(img, countour_centers):
    for cc in countour_centers:
        cv2.circle(img, cc, 5, (0, 255, 0), -1)


def project_to_ground_plane(points, homography_matrix):
    # Apply homography to project points to the ground plane
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    ground_points_homogeneous = points_homogeneous @ homography_matrix.T
    ground_points = (
        ground_points_homogeneous[:, :2] / ground_points_homogeneous[:, 2][:, np.newaxis]
    )
    return ground_points


def project_to_image_plane(points, homography_matrix):
    # Apply inverse homography to project points back to the image plane
    homography_matrix_inv = np.linalg.inv(homography_matrix)
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    image_points_homogeneous = points_homogeneous @ homography_matrix_inv.T
    image_points = image_points_homogeneous[:, :2] / image_points_homogeneous[:, 2][:, np.newaxis]
    return image_points


def calculate_homography_matrix(fov, image_width, image_height, tilt_angle, camera_height):
    # Calculate the focal length from the FOV
    focal_length = image_width / (2 * np.tan(np.deg2rad(fov) / 2))

    # Camera intrinsic matrix
    cx = image_width / 2
    cy = image_height / 2

    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

    # Rotation matrix
    theta = np.deg2rad(tilt_angle)
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

    # Translation vector
    t = np.array([0, 0, -camera_height])

    # Homography matrix
    H = K @ np.hstack((R[:, :2], t.reshape(-1, 1)))
    return H


if __name__ == "__main__":
    data = get_data()

    cv2.namedWindow("image")

    # Define the homography matrix (example values, should be calibrated for your setup)
    # homography_matrix = np.array([
    #     [1.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [0.0, 0.0, 1.0]
    # ])
    FOV = 90
    camera_tilt_angle = -15
    height__cm = 250
    homography_matrix = calculate_homography_matrix(FOV, 640, 480, camera_tilt_angle, height__cm)

    try:
        while True:
            for d in data:
                tstart = time_ns()
                # ----------------------------------

                bgr = cv2.cvtColor(d.camera_data.rgb_image, cv2.COLOR_RGB2BGR)
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

                red_hsv = red_mask_thresh(hsv)
                red_bgr = cv2.cvtColor(red_hsv, cv2.COLOR_HSV2BGR)

                contours = get_marker_contours(red_bgr)
                contour_centers = get_contour_centers(contours)
                draw_contour_centers(red_bgr, contour_centers)

                if len(contour_centers) > 1:
                    # Project contour centers to the ground plane
                    contour_centers_np = np.array(contour_centers, dtype=np.float32)
                    ground_points = project_to_ground_plane(contour_centers_np, homography_matrix)
                    ground_points_copy = ground_points.copy()

                    # Fit a polynomial through the ground points
                    x_ground = ground_points[:, 0]
                    y_ground = ground_points[:, 1]
                    poly_coeff = np.polyfit(x_ground, y_ground, 2)  # Fit a 2nd degree polynomial
                    poly = np.poly1d(poly_coeff)

                    # Generate points along the polynomial curve in ground plane
                    x_new_ground = np.linspace(x_ground.min(), x_ground.max(), 100)
                    y_new_ground = poly(x_new_ground)
                    ground_poly_points = np.vstack((x_new_ground, y_new_ground)).T

                    # Project polynomial points back to the image plane
                    image_poly_points = project_to_image_plane(
                        ground_poly_points, homography_matrix
                    )

                    # Draw the polynomial curve in the image
                    for i in range(len(image_poly_points) - 1):
                        cv2.line(
                            red_bgr,
                            (int(image_poly_points[i][0]), int(image_poly_points[i][1])),
                            (int(image_poly_points[i + 1][0]), int(image_poly_points[i + 1][1])),
                            (0, 255, 0),
                            2,
                        )

                # Concatenate the original and processed images horizontally
                combined_image = np.hstack((bgr, red_bgr))

                cv2.imshow("image", combined_image)
                key_pressed = cv2.waitKey(33) & 0xFF
                if key_pressed == ord("q"):
                    break
                elif key_pressed == ord(" "):
                    while True:
                        cv2.imshow("image", combined_image)
                        if cv2.waitKey(33) & 0xFF == ord(" "):
                            break


                # ----------------------------------
                tend = time_ns()
                dt = tstart - tend
                # print(f"{dt / 1e9:.2f}")
    finally:
        cv2.destroyAllWindows()
