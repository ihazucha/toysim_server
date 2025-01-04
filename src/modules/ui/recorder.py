from time import sleep, time_ns

from PySide6.QtCore import QThread

from modules.recorder import RecordWriter
from utils.ipc import SPMCQueue
from utils.data import record_path


class RecordingThread(QThread):
    def __init__(self, q_simulation: SPMCQueue):
        super().__init__()
        self._is_recording = False
        self._record_writer = RecordWriter(q_simulation)

    def run(self):
        while True:
            if not self._is_recording:
                sleep(0.1)
            else:
                path = record_path(str(time_ns()))
                self._record_writer.write_new(record_path=path)

    def toggle(self, is_recording: bool):
        if not is_recording:
            self._record_writer.stop()
        self._is_recording = is_recording
