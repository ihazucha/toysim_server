import socket
from queue import Queue
from threading import Event, Thread

from ToySim.settings import ClientTypes, NetworkSettings


class TcpServer:
    def __init__(
        self,
        recv_queue: Queue,
        send_queue: Queue,
        connected_event: Event,
        listen_addr:tuple=None,
        client=ClientTypes.SIMULATION,
        verbose: bool = True,
    ):
        self._recv_queue = recv_queue
        self._send_queue = send_queue
        self._connected_event = connected_event
        self._client = client
        self._verbose = verbose
        self._thread = Thread(target=self._await_connection, daemon=True)
        
        if listen_addr is None:
            listen_addr = NetworkSettings.Simulation.SERVER_ADDR if client == ClientTypes.SIMULATION else NetworkSettings.Vehicle.SERVER_ADDR
        self._sock = self._bind_listen(listen_addr)

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

    def _recv_all(self, size):
        data = b""
        while len(data) < size:
            more = self._socket.recv(size - len(data))
            if not more:
                raise IOError("Socket closed before all data received")
            data += more
        return data

    def _recv_data(self):
        return self._recv_all(self._recv_size)

    def receive_loop(self):
        exit_event = Event()
        def send_data():
            while not exit_event.is_set():
                try:
                    data = self._send_queue.get()
                    self._socket.sendall(data)
                except OSError:
                    self._log("Send failed - connection closed by client")
                    break
                    

        def recv_data():
            while True:
                try:
                    data = self._recv_data()
                    self._recv_queue.put(data)
                except OSError:
                    self._log("Recv failed - connection closed by client")
                    break
            exit_event.set()
        
        # Create and start the receiving thread
        recv_thread = Thread(target=recv_data)
        recv_thread.start()
        
        # Create and start the sending thread
        send_thread = Thread(target=send_data)
        send_thread.start()
        
        send_thread.join()
        recv_thread.join()