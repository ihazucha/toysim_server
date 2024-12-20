import struct
import math
import socket

from multiprocessing import Process
from threading import Thread
from typing import Tuple

from ToySim.utils.ipc import SPMCQueue
from ToySim.utils.image import jpg_encode
from ToySim.data import JPGImageData, RawImageData

MAX_DGRAM_SIZE = 2**16


def get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


# --------------------


class ImageDataSender(Thread):
    DGRAM_HEADER_SIZE = 64
    DATA_HEADER_SIZE = struct.calcsize("=BQ")
    MAX_DATA_SIZE = MAX_DGRAM_SIZE - DGRAM_HEADER_SIZE - DATA_HEADER_SIZE

    def __init__(self, q_image: SPMCQueue, addr: Tuple[str, int]):
        super().__init__()
        self._q_image = q_image
        self._addr = addr

    def run(self) -> None:
        q = self._q_image.get_consumer()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock = sock
            while True:
                raw_image_data: RawImageData = q.get()
                jpg = jpg_encode(raw_image_data.image_array, 80)
                self._send(jpg, raw_image_data.timestamp)

    def _send(self, image, timestamp: int):
        size = len(image)  # TODO: optimise (could be estimated and done without len4each)
        segment_count = math.ceil(size / self.__class__.MAX_DATA_SIZE)
        header = struct.pack("=BQ", segment_count, timestamp)
        image_mv = memoryview(image)
        start_pos = 0
        while segment_count:
            end_pos = min(size, start_pos + self.__class__.MAX_DATA_SIZE)
            self._sock.sendto(header + image_mv[start_pos:end_pos], self._addr)
            start_pos = end_pos
            segment_count -= 1


class ImageDataReceiver(Thread):
    def __init__(self, q_image: SPMCQueue, addr: Tuple[str, int]):
        super().__init__()
        self._q_image = q_image
        self._addr = addr

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(self._addr)
            self._sock = s
            q = self._q_image.get_producer()
            while True:
                jpg_image_data = self._recv()
                q.put(jpg_image_data)

    def _recv(self) -> JPGImageData:
        jpg_data = b""
        segment = 255
        while segment != 1:
            dgram, _ = self._sock.recvfrom(MAX_DGRAM_SIZE)
            segment, timestamp = struct.unpack("=BQ", dgram[0:9])
            jpg_data += dgram[9:]
        return JPGImageData(timestamp=timestamp, jpg=jpg_data)


# --------------------


class SensorDataSender(Thread):
    def __init__(self, q_sensor: SPMCQueue, addr: Tuple[str, int]):
        super().__init__()
        self._addr = addr
        self._q_sensor = q_sensor

    def run(self):
        q = self._q_sensor.get_consumer()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            while True:
                dataframe = q.get()
                sock.sendto(dataframe.to_bytes(), self._addr)


class SensorDataReceiver(Thread):
    def __init__(self, q_sensor: SPMCQueue, addr: Tuple[str, int]):
        super().__init__()
        self._q_sensor = q_sensor
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(addr)

    def run(self):
        try:
            self._run()
        finally:
            self._sock.close()

    def get_client_ip(self):
        _, addr = self._sock.recvfrom(MAX_DGRAM_SIZE)
        return addr[0]

    def _run(self):
        q = self._q_sensor.get_producer()
        while True:
            data, _ = self._sock.recvfrom(MAX_DGRAM_SIZE)
            q.put(data)


# --------------------


class RemoteDataSender(Thread):
    def __init__(self, q_remote: SPMCQueue, addr: Tuple[str, int]):
        super().__init__()
        self._q_remote = q_remote
        self._addr = addr

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            q = self._q_remote.get_consumer()
            while True:
                data = q.get()
                s.sendto(data.to_bytes(), self._addr)


class RemoteDataReceiver(Thread):
    def __init__(self, q_remote: SPMCQueue, addr: Tuple[str, int]):
        super().__init__()
        self._q_remote = q_remote
        self._addr = addr

    def run(self):
        q = self._q_remote.get_producer()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(self._addr)
            while True:
                data = sock.recv(MAX_DGRAM_SIZE)
                q.put(data)


# --------------------


class NetworkClient(Process):
    def __init__(
        self,
        q_image: SPMCQueue,
        q_sensor: SPMCQueue,
        q_remote: SPMCQueue,
        server_ip: str,
    ):
        super().__init__()
        self._q_image = q_image
        self._q_sensor = q_sensor
        self._q_remote = q_remote
        self._server_ip = server_ip

    def run(self):
        t_image = ImageDataSender(q_image=self._q_image, addr=(self._server_ip, 5500))
        t_sensor = SensorDataSender(q_sensor=self._q_sensor, addr=(self._server_ip, 5510))
        t_remote = RemoteDataReceiver(q_remote=self._q_remote, addr=(get_local_ip(), 5520))
        ts = [t_image, t_sensor, t_remote]
        [t.start() for t in ts]
        [t.join() for t in ts]


class NetworkServer(Process):
    def __init__(
        self,
        q_image: SPMCQueue,
        q_sensor: SPMCQueue,
        q_remote: SPMCQueue,
        server_ip: str = get_local_ip(),
    ):
        super().__init__()
        self._q_image = q_image
        self._q_sensor = q_sensor
        self._q_remote = q_remote
        self._server_ip = server_ip

    def run(self):
        t_image = ImageDataReceiver(q_image=self._q_image, addr=(self._server_ip, 5500))
        t_sensor = SensorDataReceiver(q_sensor=self._q_sensor, addr=(self._server_ip, 5510))
        t_remote = RemoteDataSender(q_remote=self._q_remote, addr=(t_sensor.get_client_ip(), 5520))
        threads = [t_image, t_sensor, t_remote]
        [t.start() for t in threads]
        [t.join() for t in threads]
