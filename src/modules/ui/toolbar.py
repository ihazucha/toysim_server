from PySide6.QtCore import Signal
from PySide6.QtWidgets import QToolBar, QWidget
from PySide6.QtGui import QIcon, QAction

from utils.paths import icon_path


class TopToolBar(QToolBar):
    record_toggled = Signal(bool)

    def __init__(self, parent: QWidget, name="TopToolBar"):
        super().__init__(name)
        self._parent = parent
        self._add_action_record()

    def _add_action_record(self):
        a = QAction(QIcon(icon_path("record")), "Record", self)
        a.setCheckable(True)
        a.setChecked(False)
        a.setShortcut("Ctrl+R")

        def rec_start():
            self._parent.setStyleSheet("QWidget#central { border: 3px solid red; } ")
            a.setIcon(QIcon(icon_path("stop")))
            self.record_toggled.emit(True)

        def rec_stop():
            self._parent.setStyleSheet("")
            a.setIcon(QIcon(icon_path("record")))
            self.record_toggled.emit(False)

        def toggle_record(state: bool):
            if state:
                rec_start()
            else:
                rec_stop()

        a.triggered.connect(toggle_record)
        self.addAction(a)
