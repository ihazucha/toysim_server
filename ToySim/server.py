import socket
import struct

from threading import Event, Thread
from multiprocessing import Process, Queue
from queue import Empty

from ToySim.settings import ClientTypes, NetworkSettings


def get_local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


MAX_DGRAM_SIZE = 2**16


class TcpServer(Process):
    def __init__(
        self,
        q_recv: Queue,
        q_send: Queue,
        e_connected: Event,
    ):
        super().__init__()
        self._q_recv = q_recv
        self._q_send = q_send
        self._e_connected = e_connected
        self._sock = self._bind_listen((get_local_ip(), 8888))

    def _log(self, msg: str):
        print(f"{self.__class__.name} {msg}")

    def _bind_listen(self, addr: tuple):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(addr)
        sock.listen()
        self._log(f"Active on: {addr}")
        return sock

    def run(self):
        while True:
            self._log(f"Waiting for connection...")
            sock, addr = self._sock.accept()
            self._log(f"Connected: {addr}")
            connection = TcpConnection(sock, self._q_recv, self._q_send)
            self._e_connected.set()
            connection.receive_loop()
            del connection
            self._log(f"Disconnected: {addr}")
            self._e_connected.clear()


class TcpConnection:
    def __init__(self, socket, q_recv, q_send):
        self._socket = socket
        self._q_recv = q_recv
        self._q_send = q_send

    def __del__(self):
        self._close()

    def _log(self, msg: str, who: str = "[Conn]"):
        print(f"{who} {msg}")

    def _close(self):
        if self._socket is None:
            return
        try:
            self._log(f"Closing cocket...")
            self._socket.shutdown(socket.SHUT_RDWR)
            self._socket.close()
            self._log(f"Socket closed")
        except OSError as e:
            self._log(f"Error while closing socket: {e}")
            pass

    def _recv_data(self, size):
        data = b""
        while len(data) < size:
            more = self._socket.recv(size - len(data))
            if not more:
                raise IOError("Socket closed before all data received")
            data += more
        return data

    def receive_loop(self):
        def send(exit_event: Event):
            while not exit_event.is_set():
                try:
                    data = self._q_send.get(timeout=1)
                    self._socket.sendall(data)
                except OSError:
                    exit_event.set()
                    self._log("Send failed - connection closed by client")
                    break
                except Empty:
                    pass

        def recv(exit_event: Event):
            while True:
                try:
                    data = self._recv_data(self._recv_size)
                    self._q_recv.put(data)
                except OSError:
                    self._log("Recv failed - connection closed by client")
                    exit_event.set()
                    break

        exit_event = Event()
        recv_thread = Thread(target=recv, args=[exit_event])
        send_thread = Thread(target=send, args=[exit_event])

        recv_thread.start()
        send_thread.start()

        recv_thread.join()
        send_thread.join()


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

    def get_client_ip(self):
        _, addr = self._sock.recvfrom(MAX_DGRAM_SIZE)
        return addr[0]

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
    def __init__(self, q_image: Queue, q_sensor: Queue, q_control: Queue):
        super().__init__()
        self._q_image = q_image
        self._q_sensor = q_sensor
        self._q_control = q_control

    def run(self):
        image_receiver = UDPImageReceiver(queue=self._q_image)
        sensor_receiver = UDPSensorReceiver(queue=self._q_sensor)
        control_sender = UDPControlSender(queue=self._q_control, addr=sensor_receiver.get_client_ip())
        threads = [image_receiver, sensor_receiver, control_sender]
        [t.start() for t in threads]
        [t.join() for t in threads]
