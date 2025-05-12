from time import sleep, time_ns

from PySide6.QtCore import QThread, Signal, Slot

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

class UIRecord:
    def __init__(self, name: str, frames_count: int, first_timestamp: int, last_timestamp: int):
        self.name = name
        self.frames_count = frames_count
        self.first_timestamp = first_timestamp
        self.last_timestamp = last_timestamp

class UIFrame:
    def __init__(self, index: int, timestamp: int):
        self.index = index
        self.timestamp = timestamp

class PlaybackThread(QThread):
    record_loaded = Signal(UIRecord)
    frame_ready = Signal(UIFrame)

    def __init__(self):
        super().__init__()
        self._is_running = False
        self._is_playing = False
        self._is_stopped = False

        self._q = messaging.q_real.get_producer()
        self._record = None
        self._record_frames = None
        self._i = 0
        self._start_i = 0
        self._end_i = 0

    def run(self):
        self._is_running = True
        while self.is_running:
            if not self._is_playing:
                sleep(0.05)
                continue
            try:
                self._play_selected_record()
            except Exception as e:
                print(e)
                self._is_playing = False

    def stop(self):
        self._is_playing = False
        self._is_stopped = True
        self._is_running = False
        self.quit()

    @property
    def is_running(self):
        return self._is_running and not self.isInterruptionRequested()
    
    @Slot(str)
    def on_record_set(self, record_name: str):
        print(f"[PB Data] on_record_set {record_name}")
        self._is_stopped = True
        self._is_playing = False
        self._i = 0
        self._load_record(record_name)
        self._play_frame(0)
    
    @Slot(int)
    def on_frame_index_set(self, index: int):
        print(f"[PB Data] on_frame_index_set {index}")
        self._i = index
        self._play_frame(index)
        
    @Slot(bool)
    def on_play_pause_toggled(self, play: bool):
        print(f"[PB Data] on_play_pause_toggle {play}")
        self._is_playing = play

    @Slot(int, int)
    def on_start_end_index_change(self, start, end):
        self._start_i = start
        self._end_i = end

    def _load_record(self, record_name: str):
        try:
            path = record_path(record_name)
            self._record_frames = RecordReader.read_all(path, ProcessedRealData)
            self._record = UIRecord(
                name=record_name,
                frames_count=len(self._record_frames),
                first_timestamp=self._record_frames[0].original.timestamp,
                last_timestamp=self._record_frames[-1].original.timestamp
            )
            self.record_loaded.emit(self._record)
            self._end_i = self._record.frames_count - 1
        except Exception as e:
            print(f"[E] [PlaybackThread] Error while loading record: {e}")

    def _play_frame(self, index: int):
        data = self._record_frames[index].original
        self._q.put(data.to_bytes())
        uiframe = UIFrame(index=self._i, timestamp=data.timestamp)
        self.frame_ready.emit(uiframe)

    def _play_selected_record(self):
        if self._record_frames is None or len(self._record_frames) < 2:
            print(f"[Playback] Record {self._record.name} is empty or invalid")
            return
        
        self._is_stopped = False
        self._i = self._start_i

        while self._i < self._end_i:
            if self._is_stopped:
                break
            while not self._is_playing:
                sleep(0.05)

            self._emit_frame(self._i)

            frame_data = self._record_frames[self._i].original
            next_frame_data = self._record_frames[self._i + 1].original
            sleep((next_frame_data.timestamp - frame_data.timestamp) / 1e9)
            
            self._i += 1
        # Last frame
        self._emit_frame(self._i)

    def _emit_frame(self, index):
        data = self._record_frames[index].original
        uiframe = UIFrame(index=index, timestamp=data.timestamp)
        self.frame_ready.emit(uiframe)
        self._q.put(data.to_bytes())


