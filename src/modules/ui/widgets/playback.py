import sys
from time import sleep, time_ns
from typing import Iterable

from PySide6.QtWidgets import (
    QWidget,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QPushButton,
    QLabel,
    QApplication,
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QKeyCombination
from datetime import datetime

from superqt import QRangeSlider


# Foos
# -----------------------------------------------------------------------------


def get_sample_timestamps():
    sample_timestamps = []
    current_time_ns = time_ns()
    frame_interval_ns = 1_000_000_000 // 30
    for i in range(300):
        sample_timestamps.append(current_time_ns)
        jitter = (i % 5 - 2) * 1_000_000
        current_time_ns += frame_interval_ns + jitter
    return sample_timestamps

# Recorder
# -----------------------------------------------------------------------------


class Frame:
    def __init__(self, timestamp: int):
        self.timestamp = timestamp


class UIFrame:
    def __init__(self, index: int, timestamp: int):
        self.index = index
        self.timestamp = timestamp


class UIRecord:
    def __init__(self, frames_count: int, first_timestamp: int, last_timestamp: int):
        self.frames_count = frames_count
        self.first_timestamp = first_timestamp
        self.last_timestamp = last_timestamp


class RecordReader(QThread):
    frame_ready = Signal(Frame)
    record_loaded = Signal(UIRecord)

    def __init__(self):
        super().__init__()
        self._record_frames: Iterable[Frame] = None
        self._i = 0
        self._is_running = False
        self._is_playing = False

    def load_record(self):
        self._record_frames = list(Frame(ts) for ts in get_sample_timestamps())
        ui_record = UIRecord(
            frames_count=len(self._record_frames),
            first_timestamp=self._record_frames[0].timestamp,
            last_timestamp=self._record_frames[-1].timestamp,
        )
        self.record_loaded.emit(ui_record)

    def run(self):
        end = len(self._record_frames) - 1
        self._is_running = True
        while self._is_running:
            self._i = 0
            while self._i < end:
                if not self._is_running:
                    break
                frame: Frame = self._record_frames[self._i]
                next_frame: Frame = self._record_frames[self._i + 1]
                next_frame_dt: float = (next_frame.timestamp - frame.timestamp) / 1e9
                self.frame_ready.emit(UIFrame(index=self._i, frame=frame))
                sleep(next_frame_dt)
                self._i += 1
                while self._is_running and not self._is_playing:
                    sleep(0.05)

    def _emit_frame_on_index(self, index: int):
        frame: Frame = self._record_frames[index]
        self.frame_ready.emit(UIFrame(index=index, frame=frame))

    @Slot(bool)
    def on_play_pause_toggle(self, play: bool):
        self._is_playing = play

    @Slot(int)
    def on_frame_index_set(self, index: int):
        print(f"on_frame_index_set {index}")
        self._i = index
        self._emit_frame_on_index(index)

    @Slot()
    def on_quit_requested(self):
        self._is_running = False


# Playback
# -----------------------------------------------------------------------------


class PlaybackWidget(QWidget):
    frame_changed = Signal(int)
    play_pause_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Playback")

        self._record: UIRecord = None
        self._current_frame: UIFrame = None

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        info_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0 / 0")
        self.timestamp_label = QLabel("Timestamp: 0")
        self.timestamp_label.hide()
        self.time_label = QLabel("Time: 00:00.000")

        info_layout.addWidget(self.timestamp_label)
        info_layout.addWidget(self.time_label)
        info_layout.addStretch()
        main_layout.addLayout(info_layout)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setValue(0)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.valueChanged.connect(self._on_slider_value_changed)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.frame_label)
        slider_layout.addWidget(self.timeline_slider)

        main_layout.addLayout(slider_layout)

        controls_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.setEnabled(False)
        self.play_pause_button.setCheckable(True)
        self.play_pause_button.toggled.connect(self._on_play_pause_toggled)

        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        self.setLayout(main_layout)

    @Slot(UIFrame)
    def on_next_frame(self, frame: UIFrame):
        self._current_frame = frame
        self._update_ui_elements()

    @Slot(UIRecord)
    def on_record_loaded(self, record: UIRecord):
        print(f"on_record_loaded {record}")
        self._record = record
        self.play_pause_button.setChecked(False)
        self.on_next_frame(frame=None)
        self._update_ui_elements()

    @Slot(bool)
    def _on_play_pause_toggled(self, checked):
        self.play_pause_toggled.emit(checked)
        self.play_pause_button.setText("Pause" if checked else "Play")

    @Slot(int)
    def _on_slider_value_changed(self, position):
        print(f"on_slider_value_changed {position}")
        self.play_pause_button.setChecked(False)
        self.play_pause_toggled.emit(False)
        self.frame_changed.emit(position)

    def _update_ui_elements(self):
        total_frames = self._record.frames_count if self._record is not None else 0
        current_frame_display = (
            self._current_frame.index + 1 if self._current_frame is not None else 0
        )

        self.frame_label.setText(f"Frame: {current_frame_display} / {total_frames}")

        self.timeline_slider.setMaximum(total_frames - 1)
        self.timeline_slider.setEnabled(True)
        self.play_pause_button.setEnabled(True)

        if self._current_frame is not None:
            self.timestamp_label.setText(
                f"Timestamp: {self._current_frame.timestamp / 1e9:.3f}"
            )
            relative_ts = self._current_frame.timestamp - self._record.first_timestamp
            time_str = datetime.fromtimestamp(relative_ts / 1e9).strftime("%M:%S.%f")[:-3]
            self.time_label.setText(
                f"Time: {time_str}"
            )
            if self.timeline_slider.value() != self._current_frame.index:
                self.timeline_slider.blockSignals(True)
                self.timeline_slider.setValue(self._current_frame.index)
                self.timeline_slider.blockSignals(False)
        else:
            print("_update_ui_elements (no frame)")
            self.timestamp_label.setText("Timestamp: --")
            self.time_label.setText("Time: --")
            self.timeline_slider.setValue(0)


class CustomWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("Demo")
        # self.setFocus()
        # self.resize(600, 150)

        self.playback_widget = PlaybackWidget()
        self.setCentralWidget(self.playback_widget)

    def keyPressEvent(self, event):
        if event.keyCombination() == QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_Right):
            print("Ctrl+RightArrow")
        return super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CustomWindow()
    record_reader = RecordReader()

    win.playback_widget.frame_changed.connect(record_reader.on_frame_index_set)
    win.playback_widget.play_pause_toggled.connect(record_reader.on_play_pause_toggle)
    record_reader.frame_ready.connect(win.playback_widget.on_next_frame)
    record_reader.record_loaded.connect(win.playback_widget.on_record_loaded)

    record_reader.load_record()
    record_reader.start()

    win.show()

    app.aboutToQuit.connect(record_reader.on_quit_requested)
    exit_code = app.exec()

    record_reader.wait()
    sys.exit(exit_code)
