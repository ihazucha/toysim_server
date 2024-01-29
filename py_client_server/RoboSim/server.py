import socket
from queue import Queue
from threading import Event, Thread

from RoboSim.settings import RECV_DATA_SIZE, SERVER_HOST, SERVER_PORT


class TcpServer:
    def __init__(
        self,
        recv_queue: Queue,
        send_queue: Queue,
        connected_event: Event,
        host: str = SERVER_HOST,
        port: int = SERVER_PORT,
        verbose: bool = True,
    ):
        self._recv_queue = recv_queue
        self._send_queue = send_queue
        self._connected_event = connected_event
        self._verbose = verbose
        self._thread = Thread(target=self._await_connection, daemon=True)
        self._sock = self._bind_listen((host, port))

    def start(self):
        self._thread.start()

    def _log(self, msg: str, who: str = "[Listener]"):
        if self._verbose:
            print(f"{who} {msg}")

    def _bind_listen(self, addr: tuple):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(addr)
        sock.listen()
        self._log(f"Active on: {addr}")
        return sock

    def _await_connection(self):
        while True:
            self._log(f"Waiting for connection...")
            socket, addr = self._sock.accept()
            self._log(f"Connected: {addr}")
            connection = TcpConnection(
                socket,
                self._recv_queue,
                self._send_queue,
                verbose=self._verbose,
            )
            self._connected_event.set()
            connection.receive_loop()
            del connection
            self._log(f"Disconnected: {addr}")
            self._connected_event.clear()


class TcpConnection:
    def __init__(self, socket, recv_queue, send_queue, verbose=True):
        self._socket = socket
        self._recv_queue = recv_queue
        self._send_queue = send_queue
        self._verbose = verbose

    def __del__(self):
        self._close()

    def _log(self, msg: str, who: str = "[Conn]"):
        if self._verbose:
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

    def _recv_all(self, size):
        data = b""
        while len(data) < size:
            more = self._socket.recv(size - len(data))
            if not more:
                raise IOError("Socket closed before all data received")
            data += more
        return data

    def _recv_data(self):
        return self._recv_all(RECV_DATA_SIZE)

    def receive_loop(self):
        while True:
            try:
                data = self._recv_data()
                self._recv_queue.put(data)
            except OSError:
                self._log("Connection closed by client")
                break