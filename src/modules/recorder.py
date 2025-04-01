import struct
from datalink.data import Serializable

from datalink.ipc import SPMCQueue

# TODO: create a data format such that all relevant (meta)data are stored:
# - type of client, configuration, settings, datetime, measurement & algorithm data

class RecordWriter:
    def __init__(self, data_queue: SPMCQueue):
        self._running = False
        self._data_queue = data_queue

    def write_new(self, file):
        q = self._data_queue.get_consumer()
        self._running = True
        with open(file, "wb") as f:
            while self._running:
                data: Serializable = q.get()
                data_bytes = data.to_bytes()
                data_bytes_size = struct.pack("=Q", len(data_bytes))
                f.write(data_bytes_size)
                f.write(data_bytes)

    def stop(self):
        self._running = False

class RecordReader:
    Q_SIZE = struct.calcsize("=Q")
    @staticmethod
    # TODO: make it so that data_cls identifier is stored within the record
    def read(file, data_cls: Serializable):
        with open(file, "rb") as f:
            data = []
            while True:
                data_size_bytes = f.read(RecordReader.Q_SIZE)
                if not data_size_bytes:
                    break
                data_size = struct.unpack("=Q", data_size_bytes)[0]
                data_bytes = f.read(data_size)
                if not data_bytes:
                    break
                data.append(data_cls.from_bytes(data_bytes))
            return data
