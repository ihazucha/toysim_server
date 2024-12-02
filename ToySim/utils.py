from multiprocessing import Lock, Manager
from typing import Any


class SharedBuffer:
    """One producer, multiple (possibly lagging couple of frames behind) consumers"""

    def __init__(self, size: int = 2):
        self._manager = Manager()
        self._size = size
        self._buffer = self._manager.list([None] * size)
        self._index = self._manager.Value("i", 0)
        self._lock = Lock()
        self._has_writer = False

    def get_writer(self):
        assert not self._has_writer, "[SharedBuffer] Attempt for more than one"
        self._has_writer = True
        return self.Writer(self._buffer, self._index, self._lock, self._size)

    def get_reader(self):
        return self.Reader(self._buffer, self._index, self._lock)

    class Writer:
        def __init__(self, buffer, index, lock, size):
            self._buffer = buffer
            self._index = index
            self._lock = lock
            self._size = size

        def write(self, x: Any):
            with self._lock:
                self._index.value = (self._index.value + 1) % self._size
                self._buffer[self._index.value] = x

    class Reader:
        def __init__(self, buffer, index, lock):
            self._buffer = buffer
            self._index = index
            self._lock = lock

        def head(self):
            with self._lock:
                return self._buffer[self._index.value]

        @property
        def index(self):
            return self._index.value
