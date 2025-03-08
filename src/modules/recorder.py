import struct
from modules.messaging import messaging
from datalink.data import SimData


class RecordWriter:
    def __init__(self):
        self._running = False

    def write_new(self, record_path: str):
        # TODO: dinstinguish between simulation and real car
        q = messaging.q_simulation.get_consumer()
        self._running = True
        with open(record_path, "wb") as f:
            frame_size = struct.pack("=Q", SimData.SIZE)
            f.write(frame_size)
            while self._running:
                data = q.get()
                f.write(data)

    def stop(self):
        self._running = False


class RecordReader:
    @staticmethod
    def read(record_path: str):
        with open(record_path, "rb") as f:
            frame_size = struct.unpack("=Q", f.read(struct.calcsize("=Q")))[0]
            data = []
            while True:
                frame_bytes = f.read(frame_size)
                if not frame_bytes:
                    break
                data.append(SimData.from_bytes(frame_bytes))
            return data
