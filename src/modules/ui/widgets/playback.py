import sys

sys.path.append("C:/Users/ihazu/Desktop/projects/toysim_server/src")

from time import sleep, time_ns
from typing import Iterable
from enum import IntEnum

from modules.ui.presets import COMBOBOX_STYLE, UIColors

from PySide6.QtWidgets import (
    QWidget,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSlider,
    QPushButton,
    QLabel,
    QApplication,
    QComboBox,
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QKeyCombination, QSize
from datetime import datetime

from superqt import QLabeledRangeSlider, QEnumComboBox
from superqt.fonticon import icon
from fonticon_mdi7 import MDI7

from PySide6.QtCore import QDir
from utils.paths import PATH_STATIC

QDir.addSearchPath("icons", PATH_STATIC)


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
                self.frame_ready.emit(UIFrame(index=self._i, timestamp=frame.timestamp))
                sleep(next_frame_dt)
                self._i += 1
                while self._is_running and not self._is_playing:
                    sleep(0.05)

    def _emit_frame_on_index(self, index: int):
        frame: Frame = self._record_frames[index]
        self.frame_ready.emit(UIFrame(index=index, timestamp=frame.timestamp))

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
    start_end_changed = Signal(int, int)
    play_pause_toggled = Signal(bool)
    repeat_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Playback")

        self._record: UIRecord = None
        self._current_frame: UIFrame = None

        self._init_ui()

    def _init_ui(self):
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setContentsMargins(0, 0, 3, 0)
        self.timeline_slider.valueChanged.connect(self._on_slider_value_changed)

        self.timeline_start_end_slider = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self.timeline_start_end_slider.setValue((0, 0))
        self.timeline_start_end_slider.hide()
        self.timeline_start_end_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.timeline_start_end_slider.setBarMovesAllHandles(False)
        self.timeline_start_end_slider.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode.NoLabel)
        # create bit of a margin for timeline_start_end_slider
        self.timeline_start_end_slider.setContentsMargins(0, 0, 3, 0)
        self.timeline_start_end_slider.setHandleLabelPosition(
            QLabeledRangeSlider.LabelPosition.LabelsOnHandle
        )
        self.timeline_start_end_slider.label_shift_y = 18
        self.timeline_start_end_slider.label_shift_x = -1
        self.timeline_start_end_slider.valueChanged.connect(self._on_start_end_values_changed)

        self.play_icon = icon(MDI7.play_outline, color=UIColors.ON_PRIMARY, scale_factor=1.05)
        self.pause_icon = icon(MDI7.pause, color=UIColors.ON_PRIMARY)
        self.repeat_on_icon = icon(MDI7.repeat, color=UIColors.ON_PRIMARY)
        self.repeat_off_icon = icon(MDI7.repeat_off, color=UIColors.ON_PRIMARY)
        self.config_icon = icon(MDI7.tune, color=UIColors.ON_PRIMARY, scale_factor=0.85)

        self.play_pause_button = QPushButton()
        self.play_pause_button.setFlat(True)
        self.play_pause_button.setIconSize(QSize(36, 36))
        self.play_pause_button.setIcon(self.play_icon)
        self.play_pause_button.setCheckable(True)
        self.play_pause_button.toggled.connect(self._on_play_pause_toggled)

        self.repeat_button = QPushButton()
        self.repeat_button.setFlat(True)
        self.repeat_button.setIconSize(QSize(36, 36))
        self.repeat_button.setIcon(self.repeat_off_icon)
        self.repeat_button.setCheckable(True)
        self.repeat_button.setChecked(True)
        self.repeat_button.toggled.connect(self._on_repeat_toggled)

        self.config_button = QPushButton()
        self.config_button.setFlat(True)
        self.config_button.setIconSize(QSize(36, 36))
        self.config_button.setIcon(self.config_icon)
        self.config_button.setCheckable(True)
        self.config_button.setChecked(False)
        self.config_button.toggled.connect(self._on_config_toggled)

        self.frame_indicator_select = QComboBox()
        self.frame_indicator_select.setStyleSheet(COMBOBOX_STYLE)
        self.frame_indicator_select.setMinimumWidth(155)
        self.frame_indicator_select.setFixedHeight(42)
        self.frame_indicator_select.addItem("00:00.000 / 00:00.000")
        self.frame_indicator_select.addItem("0 / 0")

        layout = QGridLayout(self)
        cm = layout.contentsMargins()
        cm.setTop(2)
        layout.setContentsMargins(cm)

        layout.addWidget(self.repeat_button, 0, 0)
        layout.addWidget(self.play_pause_button, 0, 1)
        layout.addWidget(self.timeline_slider, 0, 2)
        layout.addWidget(self.timeline_start_end_slider, 0, 2)
        layout.addWidget(self.frame_indicator_select, 0, 3)
        layout.addWidget(self.config_button, 0, 4)

        self.setLayout(layout)

    @Slot(UIFrame)
    def on_next_frame(self, frame: UIFrame):
        self._current_frame = frame
        self._update_ui_elements()

    @Slot(UIRecord)
    def on_record_loaded(self, record: UIRecord):
        self._record = record
        self.play_pause_button.setChecked(False)
        self._reset_ui_elements()

    @Slot(bool)
    def _on_play_pause_toggled(self, play):
        self.play_pause_toggled.emit(play)
        self.play_pause_button.setIcon(self.pause_icon if play else self.play_icon)

    @Slot(bool)
    def _on_repeat_toggled(self, repeat):
        self.repeat_toggled.emit(repeat)
        self.repeat_button.setIcon(self.repeat_off_icon if repeat else self.repeat_on_icon)

    @Slot(bool)
    def _on_config_toggled(self, checked):
        self.timeline_start_end_slider.show() if checked else self.timeline_start_end_slider.hide()

    @Slot(int)
    def _on_slider_value_changed(self, position):
        self.play_pause_button.setChecked(False)
        self.play_pause_toggled.emit(False)
        self.frame_changed.emit(position)

    @Slot(tuple)
    def _on_start_end_values_changed(self, start_end: tuple):
        # Skip default value change
        if self._current_frame is None:
            return
        start, end = start_end
        if self._current_frame.index < start:
            self.timeline_slider.setValue(start)
        elif self._current_frame.index > end:
            self.timeline_slider.setValue(end)
        self.start_end_changed.emit(start, end)

    def _reset_ui_elements(self):
        slider_max = self._record.frames_count - 1
        self.timeline_slider.setValue(0)
        self.timeline_slider.setMaximum(slider_max)
        self.timeline_start_end_slider.setMaximum(slider_max)
        self.timeline_start_end_slider.setValue((0, slider_max))
        self.play_pause_button.setEnabled(True)


    def _update_ui_elements(self):
        assert self._record is not None, "Record should be set here"
        assert self._current_frame is not None, "Current frame should be set here"

        t = self._timestamp_to_relative_time(self._current_frame.timestamp)
        end_t = self._timestamp_to_relative_time(self._record.last_timestamp)
        self.frame_indicator_select.setItemText(0, f"{t} / {end_t}")

        i = self._current_frame.index
        last_i = self._record.frames_count - 1
        self.frame_indicator_select.setItemText(1, f"{i} / {last_i}")

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(i)
        self.timeline_slider.blockSignals(False)

    def _timestamp_to_time(self, timestamp_ns: int):
        return datetime.fromtimestamp(timestamp_ns / 1e9).strftime("%M:%S.%f")[:-3]

    def _timestamp_to_relative_time(self, timestamp: int):
        assert self._record is not None, "Record should be set here"
        relative_ts = timestamp - self._record.first_timestamp
        return self._timestamp_to_time(relative_ts)


class CustomWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("Demo")
        self.setFocus()
        self.setMinimumWidth(600)

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
