import socket
from queue import Queue, Empty
from threading import Event, Thread

from ToySim.settings import ClientTypes, NetworkSettings

def get_local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]

class TcpServer:
    def __init__(
        self,
        recv_queue: Queue,
        send_queue: Queue,
        connected_event: Event,
        address: tuple,
        client: ClientTypes,
        verbose: bool = True,
    ):
        self._recv_queue = recv_queue
        self._send_queue = send_queue
        self._connected_event = connected_event
        self._client = client
        self._verbose = verbose
        self._thread = Thread(target=self._await_connection, daemon=True)
        self._sock = self._bind_listen(address)
        
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
                client=self._client,
                verbose=self._verbose,
            )
            self._connected_event.set()
            connection.receive_loop()
            del connection
            self._log(f"Disconnected: {addr}")
            self._connected_event.clear()


class TcpConnection:
    def __init__(self, socket, recv_queue, send_queue, client=ClientTypes.SIMULATION, verbose=True):
        self._socket = socket
        self._recv_queue = recv_queue
        self._send_queue = send_queue
        self._client = client
        self._verbose = verbose

        if self._client == ClientTypes.SIMULATION:
            self._recv_size = NetworkSettings.Simulation.RECV_DATA_SIZE_BYTES
        elif self._client == ClientTypes.VEHICLE:
            self._recv_size = NetworkSettings.Vehicle.RECV_DATA_SIZE_BYTES

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

    def _recv_data(self, size):
        data = b""
        while len(data) < size:
            more = self._socket.recv(size - len(data))
            if not more:
                raise IOError("Socket closed before all data received")
            data += more
        return data

    def receive_loop(self):
        def send(exit_event:Event):
            while not exit_event.is_set():
                try:
                    data = self._send_queue.get(timeout=1)
                    self._socket.sendall(data)
                except OSError:
                    exit_event.set()
                    self._log("Send failed - connection closed by client")
                    break
                except Empty:
                    pass
                    
        def recv(exit_event:Event):
            while True:
                try:
                    data = self._recv_data(self._recv_size)
                    self._recv_queue.put(data)
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
        