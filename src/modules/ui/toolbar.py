from PySide6.QtCore import Signal
from PySide6.QtWidgets import QToolBar, QWidget, QStyle, QWidgetAction, QHBoxLayout, QSpacerItem, QSizePolicy
from PySide6.QtGui import QIcon, QAction

from utils.paths import icon_path


class TopToolBar(QToolBar):
    record_toggled = Signal(bool)
    config_toggled = Signal(bool)
    sidebar_toggled = Signal(bool)

    def __init__(self, parent: QWidget, name="TopToolBar"):
        super().__init__(name)
        self._parent = parent
        self._init_layout()
        self._add_action_toggle_sidebar()
        self._add_action_record()
        self._add_action_toggle_config()

    def _init_layout(self):
        left_spacer_widget = QWidget()
        left_spacer_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        left_spacer_action = QWidgetAction(self)
        left_spacer_action.setDefaultWidget(left_spacer_widget)

        middle_spacer_widget = QWidget()
        middle_spacer_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        middle_spacer_action = QWidgetAction(self)
        middle_spacer_action.setDefaultWidget(middle_spacer_widget)

        right_spacer_widget = QWidget()
        right_spacer_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        right_spacer_action = QWidgetAction(self)
        right_spacer_action.setDefaultWidget(right_spacer_widget)

        self.addAction(left_spacer_action)
        self.addAction(middle_spacer_action)
        self.addAction(right_spacer_action)

        self.left_spacer_action = left_spacer_action
        self.middle_spacer_action = middle_spacer_action
        self.right_spacer_action = right_spacer_action

    def _add_action_toggle_config(self):
        icon = self.style().standardIcon(QStyle.SP_FileDialogDetailedView)
        a = QAction(icon, "Config", self)
        a.setCheckable(True)
        a.toggled.connect(self.config_toggled.emit)
        a.setShortcut("Ctrl+C")
        self.insertAction(self.right_spacer_action, a)

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
        self.insertAction(self.middle_spacer_action, a)

    def _add_action_toggle_sidebar(self):
        icon = self.style().standardIcon(QStyle.SP_DirIcon)
        a = QAction(icon, "Sidebar", self)
        a.setCheckable(True)
        a.toggled.connect(self.sidebar_toggled.emit)
        a.setShortcut("Ctrl+E")
        self.insertAction(self.left_spacer_action, a)
