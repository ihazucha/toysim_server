import cv2

# Path to the H.264 file
video_file = 'received_video.h264'  # Change this to your file path

# Create a VideoCapture object
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    print(frame)
    if not ret:
        print("End of video or error reading frame.")
        break  # Exit the loop if no frame is returned

    # Display the resulting frame
    cv2.imshow('H.264 Video Playback', frame)

    # Press 'q' on the keyboard to exit the playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()