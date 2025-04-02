from time import sleep, time_ns

from PySide6.QtCore import QThread

from modules.recorder import RecordReader, RecordWriter
from datalink.ipc import SPMCQueue
from datalink.data import ProcessedRealData
from utils.paths import record_path

from modules.messaging import messaging

class RecorderThread(QThread):
    def __init__(self, data_queue: SPMCQueue):
        super().__init__()
        self._is_running = False
        self._is_recording = False
        self._record_writer = RecordWriter(data_queue)

    def run(self):
        self._is_running = True
        while self._is_running and not self.isInterruptionRequested():
            if not self._is_recording:
                sleep(0.1)
            else:
                path = record_path(str(time_ns()))
                self._record_writer.write_new(file=path)

    def stop(self):
        self._is_running = False
        self.quit()

    def toggle(self, is_recording: bool):
        self._is_recording = is_recording
        if not is_recording:
            self._record_writer.stop()

class PlaybackThread(QThread):
    def __init__(self, data_queue: SPMCQueue):
        super().__init__()
        self.data_queue = data_queue
        self._is_running = False
        self._is_playing = False
        self._is_stopped = False
        self._selected_record = None

    def run(self):
        self._is_running = True
        while self.is_running:
            try:
                if self._is_playing:
                    self._is_stopped = False
                    self._play_selected_record()
                sleep(0.05)
            except Exception as e:
                print(e)
                self._is_playing = False

    def stop(self):
        self._is_playing = False
        self._is_stopped = True
        self._is_running = False
        self.quit()

    def set_current_record(self, record_name: str):
        self._selected_record = record_name
        self._is_stopped = self._is_playing
        self.toggle(False) 

    def toggle(self, is_playing: bool):
        self._is_playing = is_playing

    def _play_selected_record(self):   
        path = record_path(self._selected_record)
        data = RecordReader.read_all(path, ProcessedRealData)
        if data is None or len(data) < 2:
            print(f"[Playback] Record {self._selected_record} is empty or invalid")
            return
        
        q = messaging.q_real.get_producer()
        
        first_dt = data[1].original.timestamp - data[0].original.timestamp
        last_timestamp = data[0].original.timestamp - first_dt

        frame_count = len(data)
        i = 0
        while i < frame_count:
            if self._is_stopped:
                break
            if not self._is_playing:
                sleep(0.05)
                continue

            q.put(data[i].original.to_bytes())
            sleep((data[i].original.timestamp - last_timestamp) / 1e9)
            last_timestamp = data[i].original.timestamp
            i += 1

    @property
    def is_running(self):
        return self._is_running and not self.isInterruptionRequested()