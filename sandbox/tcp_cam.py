import sys
import cv2
import numpy as np
import argparse
import socket
import struct
from multiprocessing import Process, Queue

MAX_DGRAM = 2**16

parser = argparse.ArgumentParser(description="")
parser.add_argument("-a", "--addr", type=str, default="localhost", help="Bind port")
parser.add_argument("-p", "--port", type=int, default="5555", help="Bind address")
args = parser.parse_args()

class UDPImageServer:
    MAX_DGRAM_SIZE = 2**16

    def __init__(self, addr: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((addr, port))

    def recv_image(self) -> tuple[bytes, int]:
        image = b""
        segment = 255
        while segment != 1:
            dgram = self._sock.recv(self.__class__.MAX_DGRAM_SIZE)
            segment, timestamp = struct.unpack("=BQ", dgram[0:9])
            image += dgram[9:]
        return (image, timestamp)

def udp_server(queue: Queue, addr: str, port: int):
    udp = UDPImageServer(addr=addr, port=port)
    while True:
        jpg_data, timestamp = udp.recv_image()
        queue.put((jpg_data, timestamp))

def visualizer(queue: Queue):
    prev_timestamp = 0
    while True:
        jpg_data, timestamp = queue.get()
        try:
            image = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[Server] JPG decode problem: {e}")
            continue
        text = f"dt: {timestamp - prev_timestamp} [ms]"
        cv2.putText(
            image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        prev_timestamp = timestamp
        cv2.imshow("UDPCam", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    frame_queue:Queue = Queue(maxsize=1)
    p_udp = Process(target=udp_server, args=(frame_queue, args.addr, args.port))
    p_viz = Process(target=visualizer, args=(frame_queue,))
    
    p_udp.start()
    p_viz.start()
    
    try:
        p_udp.join()
        p_viz.join()
    except (KeyboardInterrupt, SystemExit):
        p_udp.terminate()
        p_viz.terminate()
        p_udp.join()
        p_viz.join()
        sys.exit()
