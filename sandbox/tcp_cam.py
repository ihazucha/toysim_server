import math
import sys
import cv2
import numpy as np
import argparse
import socket
import struct
from multiprocessing import Process, Queue
from threading import Thread
import time

MAX_DGRAM_SIZE = 2**16


def get_local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


class UDPImageReceiver(Thread):
    def __init__(self, queue: Queue, addr: str = get_local_ip(), port: int = 5500):
        super().__init__()
        self._queue = queue
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((addr, port))

    def _recv(self) -> tuple[bytes, int]:
        image = b""
        segment = 255
        while segment != 1:
            dgram, _ = self._sock.recvfrom(MAX_DGRAM_SIZE)
            segment, timestamp = struct.unpack("=BQ", dgram[0:9])
            image += dgram[9:]
        return (image, timestamp)

    def stop(self):
        self._sock.close()

    def run(self):
        try:
            self._run()
        finally:
            self.close()

    def _run(self):
        while True:
            jpg_data, timestamp = self._recv()
            self._queue.put((jpg_data, timestamp))


class UDPSensorReceiver(Thread):
    def __init__(self, queue: Queue, addr: str = get_local_ip(), port: int = 5510):
        super().__init__()
        self._queue = queue
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((addr, port))

    def stop(self):
        self._sock.close()

    def run(self):
        try:
            self._run()
        finally:
            self.stop()

    def _run(self):
        while True:
            data, _ = self._sock.recvfrom(MAX_DGRAM_SIZE)
            self._queue.put(data)


class UDPControlSender(Thread):
    def __init__(self, queue: Queue, addr: str, port: int = 5520):
        super().__init__()
        self._queue = queue
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.connect((addr, port))

    def stop(self):
        self._sock.close()

    def run(self):
        try:
            self._run()
        finally:
            self.stop()

    def _run(self):
        while True:
            data = self._queue.get()
            self._sock.send(data)


class Network(Process):
    def __init__(self, q_image: Queue, q_sensor: Queue, q_control: Queue, addr: str):
        super().__init__()
        self._q_image = q_image
        self._q_sensor = q_sensor
        self._q_control = q_control
        self._addr = addr

    def run(self):
        image_receiver = UDPImageReceiver(queue=self._q_image)
        sensor_receiver = UDPSensorReceiver(queue=self._q_sensor)
        control_sender = UDPControlSender(queue=self._q_control, addr=self._addr)
        threads = [image_receiver, sensor_receiver, control_sender]
        [t.start() for t in threads]
        [t.join() for t in threads]




class Generator(Process):
    def __init__(self, queue: Queue):
        super().__init__()
        self._queue = queue

    def run(self):
        t = 0.0
        while True:
            x = math.sin(t)
            t += 0.05
            set_steering_angle, set_speed = x, x
            timestamp = time.time()
            data = struct.pack("3d", timestamp, set_steering_angle, set_speed)
            self._queue.put(data)
            time.sleep(0.05)


class Visualizer(Process):
    def __init__(self, q_image: Queue, q_sensor: Queue) -> None:
        super().__init__()
        self._q_image = q_image
        self._q_sensor = q_sensor

    def close(self):
        cv2.destroyAllWindows()

    def run(self):
        try:
            self._run()
        finally:
            self.close

    def _run(self):
        prev_timestamp = 0
        while True:
            jpg_data, timestamp = self._q_image.get()
            data = self._q_sensor.get()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # TODO: remove this and use the first receive address instead
    parser.add_argument("-a", "--addr", type=str, default="192.168.0.103", help="Client address")
    args = parser.parse_args()

    q_image: Queue = Queue(maxsize=1)
    q_sensor: Queue = Queue(maxsize=1)
    q_control: Queue = Queue(maxsize=1)

    p_network = Network(q_image=q_image, q_sensor=q_sensor, q_control=q_control, addr=args.addr)
    p_generator = Generator(queue=q_control)
    p_visualizer = Visualizer(q_image, q_sensor)

    proceses = [p_network, p_generator, p_visualizer]
    [p.start() for p in proceses]  # type: ignore

    try:
        [p.join() for p in proceses]  # type: ignore
    except (KeyboardInterrupt, SystemExit):
        print("[Server] Interrupted, exiting...")
    finally:
        [p.terminate() for p in proceses]  # type: ignore
        [p.join() for p in proceses]  # type: ignore
        sys.exit()
