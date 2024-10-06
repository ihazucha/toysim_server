import socket
import cv2
import numpy as np
import av

# Configuration
SERVER_IP = "192.168.0.104"  # Replace with the server's IP address
SERVER_PORT = 10001          # Same port as the server

def decode_stream(sock):
    # Wrap the socket in a file-like object
    sock_file = sock.makefile('rb')
    container = av.open(sock_file, format='h264')

    for frame in container.decode(video=0):
        # Convert frame to numpy array
        img = frame.to_ndarray(format='bgr24')
        
        # Display the frame using OpenCV
        cv2.imshow("H.264 Video Playback", img)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    # Create a TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to the server
        sock.connect((SERVER_IP, SERVER_PORT))

        print("Connected to server. Receiving data...")
        
        # Decode and display the video stream
        decode_stream(sock)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()