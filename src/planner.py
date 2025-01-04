from copy import deepcopy
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
    cv2.createTrackbar('HMin','image',0,179,lambda: ...) # 0-179 for Opencv
    cv2.createTrackbar('HMax','image',0,179,lambda: ...)
    cv2.createTrackbar('SMin','image',0,255,lambda: ...)
    cv2.createTrackbar('SMax','image',0,255,lambda: ...)
    cv2.createTrackbar('VMin','image',0,255,lambda: ...)
    cv2.createTrackbar('VMax','image',0,255,lambda: ...)

    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

# HMax set to 12
def hsv_thresh(hsv):
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')
    
    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(bgr, bgr, mask = mask)
    return output

def red_mask_thresh(hsv):
    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([7,255,255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([172,50,50])
    upper_red = np.array([179,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    result = cv2.bitwise_and(hsv, hsv, mask=mask) 
    # result[np.where(mask == 0)] = 0
    return result

if __name__ == "__main__":
    data = get_data()

    cv2.namedWindow("image")
    hsv_thresh_sliders()

    try:
        while True:
            for d in data:
                tstart = time_ns()
                # ----------------------------------
            
                bgr = cv2.cvtColor(d.camera_data.rgb_image, cv2.COLOR_RGB2BGR)
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

                # red_filtered = hsv_thresh()
                red_filtered = cv2.cvtColor(red_mask_thresh(hsv), cv2.COLOR_HSV2BGR)

                # Convert the filtered image to grayscale
                gray = cv2.cvtColor(red_filtered, cv2.COLOR_BGR2GRAY)
                
                # Find contours in the grayscale image
                contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Filter out small contours by area
                min_contour_area = 10 # You can adjust this threshold
                contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
                
                # Loop over the contours
                for contour in contours:
                    # Get the moments to calculate the center of the contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # Draw a green circle at the center
                        cv2.circle(red_filtered, (cX, cY), 5, (0, 255, 0), -1)

                # Collect contour centers
                contour_centers = []
                for contour in contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        contour_centers.append((cX, cY))

                if len(contour_centers) > 1:
                    # Fit a polynomial through the contour centers
                    contour_centers = np.array(contour_centers, dtype=np.float32)
                    x = contour_centers[:, 0]
                    y = contour_centers[:, 1]
                    poly_coeff = np.polyfit(x, y, 2)  # Fit a 2nd degree polynomial
                    poly = np.poly1d(poly_coeff)

                    # Generate points along the polynomial curve
                    x_new = np.linspace(x.min(), x.max(), 100)
                    y_new = poly(x_new)

                    # Draw the polynomial curve
                    for i in range(len(x_new) - 1):
                        cv2.line(red_filtered, (int(x_new[i]), int(y_new[i])), (int(x_new[i+1]), int(y_new[i+1])), (0, 255, 0), 2)

                cv2.imshow('image', red_filtered)
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break

                # ----------------------------------
                tend = time_ns()
                dt = tstart - tend
                print(f"{dt / 1e9:.2f}")
    finally:
        cv2.destroyAllWindows()