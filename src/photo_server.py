import zmq
import cv2
import numpy as np
import argparse
import os

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser(description='Camera client connects to tcp://--ip:--port and displays live camera feed. Press <Spacebar> to save image to --save-dir directory and <Esc> to quit.')
    parser.add_argument('--ip', help='Connct to IP address')
    parser.add_argument('--port', type=int, default=5001, help='Connect to port (default: 5001)')
    parser.add_argument('--save-dir', default=os.getcwd(), help=f'Directory to save images (default: {os.getcwd()})')
    args = parser.parse_args()

    # Camera connection
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{args.ip}:{args.port}")
    socket.subscribe('')
    print(f"Subscribed to tcp://{args.ip}:{args.port}")

    cv2.namedWindow('RPi Camera', cv2.WINDOW_NORMAL)

    while True:
        # Get JPG
        jpg_bytes = socket.recv()
        jpg = np.frombuffer(jpg_bytes, dtype=np.uint8)
        img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)

        # Resize to window size 
        h, w = img.shape[:2]
        window_rect = cv2.getWindowImageRect('RPi Camera')
        if window_rect[2] > 0 and window_rect[3] > 0:
            window_w, window_h = window_rect[2], window_rect[3]
            scale = min(window_w / w, window_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_img = cv2.resize(img, (new_w, new_h))
            
            # Create black canvas and center the image
            canvas = np.zeros((window_h, window_w, 3), dtype=np.uint8)
            x_offset = (window_w - new_w) // 2
            y_offset = (window_h - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
            cv2.imshow('RPi Camera', canvas)
        else:
            cv2.imshow('RPi Camera', img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break
        elif key == 32:  # Spacebar to save
            fname = os.path.join(args.save_dir, "image.jpg")
            cv2.imwrite(fname, img)
            print(f"Saved {fname}")
    
    cv2.destroyAllWindows()
