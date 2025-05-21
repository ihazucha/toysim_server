import struct
from datalink.data import Serializable

from datalink.ipc import AbstractQueue

# TODO: create a data format such that all relevant (meta)data are stored:
# - type of client, configuration, settings, datetime, measurement & algorithm data

class RecordWriter:
    def __init__(self, recorded_queue: AbstractQueue):
        self._running = False
        self._q = recorded_queue

    def write_new(self, file):
        q = self._q.get_consumer()
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
    def read_all(file, data_cls: Serializable):
        with open(file, "rb") as f:
            data = []
            while True:
                frame = RecordReader._read_one(f, data_cls)
                if frame is None:
                    break
                data.append(frame)
            return data

    @staticmethod
    def read_one(file, data_cls: Serializable) -> Serializable | None:
        with open(file, "rb") as f:
            return RecordReader._read_one(f, data_cls)
    
    @staticmethod
    def _read_one(file_handle, data_cls: Serializable) -> Serializable | None:
        frame_size_bytes = file_handle.read(RecordReader.Q_SIZE)
        if not frame_size_bytes:
            return None
        
        frame_size = struct.unpack("=Q", frame_size_bytes)[0]
        data_bytes = file_handle.read(frame_size)
        if not data_bytes:
            return None
        
        return data_cls.from_bytes(data_bytes)