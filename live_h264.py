import socket
import cv2
import numpy as np
import av

# Configuration
SERVER_IP = "192.168.0.105"  # Replace with the server's IP address
SERVER_PORT = 5556          # Same port as the server

def decode_stream(sock):
    # Wrap the socket in a file-like object
    sock_file = sock.makefile('rb')
    container = av.open(sock_file, format='h264')

    for frame in container.decode(video=0):
        # Convert frame to numpy array
        img = frame.to_ndarray(format='bgr24')
        print(f'{img.shape}')
        
        # Display the frame using OpenCV
        cv2.imshow("H.264 Video Playback", img)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    # Create a TCP socket
    while True:
        # Connect to the server
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((SERVER_IP, SERVER_PORT))
                print("Connected to server. Receiving data...")
                decode_stream(sock)
        except KeyboardInterrupt:
            print("Interrupted, finishing...")
            break
        except Exception:
            print("Server down")
        finally:
            cv2.destroyAllWindows()


RTSP_URL = "rtsp://192.168.0.105:8554/mystream"

def rstp_main():
    # Open the RTSP stream
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    print("Connected to RTSP server. Receiving data...")

    while True:
        # Read frame from the RTSP stream
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame using OpenCV
        cv2.imshow("RTSP Video Playback", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # rstp_main()
    main()