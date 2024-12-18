import zmq

from typing import Any
from multiprocessing import Value


class SPMCQueue:
    """Single Producer Multiple Consumers Queue using ZMQ IPC sockets"""

    ZMQ_CONTEXT = zmq.Context()

    def __init__(self, port: int):
        self._port = port
        self._has_producer = Value("b", False)

    def get_producer(self):
        assert not self._has_producer.value, f"[{self.__class__.__name__}] Producer already exists"
        self._has_producer.value = True
        return SPMCQueue.Producer(self._port)

    def get_consumer(self):
        return SPMCQueue.Consumer(self._port)

    class Producer:
        def __init__(self, port: int):
            self._port = port
            self._socket = SPMCQueue.ZMQ_CONTEXT.socket(zmq.PUB)
            self._socket.bind(f"tcp://*:{self._port}")

        def __del__(self):
            if self._socket and not self._socket.closed:
                self._socket.close()

        def put(self, data: Any):
            self._socket.send_pyobj(data)

    class Consumer:
        def __init__(self, port: int):
            self._port = port
            self._socket = SPMCQueue.ZMQ_CONTEXT.socket(zmq.SUB)
            self._socket.connect(f"tcp://localhost:{self._port}")
            self._socket.subscribe("")

        def __del__(self):
            if self._socket and not self._socket.closed:
                self._socket.close()

        def get(self) -> Any:
            return self._socket.recv_pyobj()
