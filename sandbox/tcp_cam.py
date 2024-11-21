import sys
import cv2
import numpy as np
import argparse
import socket
import struct
from multiprocessing import Process, Queue

MAX_DGRAM = 2**16

def get_local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]

parser = argparse.ArgumentParser(description="")
parser.add_argument("-a", "--addr", type=str, default=get_local_ip(), help="Bind port")
parser.add_argument("-cp", "--cameraport", type=int, default="5555", help="Bind address")
parser.add_argument("-sp", "--sensorport", type=int, default="6666", help="Bind address")
args = parser.parse_args()

MAX_DGRAM_SIZE = 2**16


class UDPImageReceiver:
    def __init__(self, addr: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((addr, port))

    def recv(self) -> tuple[bytes, int]:
        image = b""
        segment = 255
        while segment != 1:
            dgram = self._sock.recv(MAX_DGRAM_SIZE)
            segment, timestamp = struct.unpack("=BQ", dgram[0:9])
            image += dgram[9:]
        return (image, timestamp)

    def close(self):
        self._sock.close()


class UDPSensorReceiver:
    def __init__(self, addr: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((addr, port))

    def recv(self) -> bytes:
        data = self._sock.recv(MAX_DGRAM_SIZE)
        return data

    def close(self):
        self._sock.close()


def camera_server(queue: Queue, addr: str, port: int):
    udp = UDPImageReceiver(addr=addr, port=port)
    try:
        while True:
            jpg_data, timestamp = udp.recv()
            queue.put((jpg_data, timestamp))
    finally:
        udp.close()


def sensor_server(queue: Queue, addr: str, port: int):
    udp = UDPSensorReceiver(addr=addr, port=port)
    try:
        while True:
            data = udp.recv()
            queue.put(data)
    finally:
        udp.close()


def visualizer(camera_queue: Queue, sensor_queue: Queue) -> None:
    prev_timestamp = 0
    while True:
        jpg_data, timestamp = camera_queue.get()
        data = sensor_queue.get()
        print(data)
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
    camera_queue: Queue = Queue(maxsize=1)
    sensor_queue: Queue = Queue(maxsize=1)
    p_camera = Process(target=camera_server, args=(camera_queue, args.addr, args.cameraport))
    p_sensor = Process(target=sensor_server, args=(sensor_queue, args.addr, args.sensorport))
    p_visualiser = Process(target=visualizer, args=(camera_queue, sensor_queue))

    p_camera.start()
    p_visualiser.start()
    p_sensor.start()

    try:
        p_camera.join()
        p_visualiser.join()
        p_sensor.join()
    except (KeyboardInterrupt, SystemExit):
        print("[Server] Interrupted, exiting...")
    finally:
        p_camera.terminate()
        p_visualiser.terminate()
        p_sensor.terminate()
        p_camera.join()
        p_visualiser.join()
        p_sensor.join()
        sys.exit()
