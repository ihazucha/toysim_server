from PySide6.QtCore import Signal, Slot, QSize
from PySide6.QtWidgets import QToolBar, QWidget, QWidgetAction, QSizePolicy
from PySide6.QtGui import QAction

from modules.ui.presets import UIColors

from superqt.fonticon import icon
from fonticon_mdi7 import MDI7


class TopToolBar(QToolBar):
    records_panel_toggled = Signal(bool)
    record_toggled = Signal(bool)
    control_panel_toggled = Signal(bool)

    def __init__(self, parent: QWidget, name="Toolbar"):
        super().__init__(name)
        self._parent = parent
        self._init_layout()

    @Slot()
    def on_record_selected(self):
        self.action_toggle_source.setChecked(True)

    def _init_layout(self):
        self.setIconSize(QSize(36, 36))
        self.setMovable(False)
        self.setStyleSheet(
            """
            QToolBar {
                border: none;
            }
        """
        )

        left_spacer = QWidget()
        left_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        left_group = QWidgetAction(self)
        left_group.setDefaultWidget(left_spacer)

        middle_spacer = QWidget()
        middle_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        middle_group = QWidgetAction(self)
        middle_group.setDefaultWidget(middle_spacer)

        right_spacer = QWidget()
        right_spacer.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        right_group = QWidgetAction(self)
        right_group.setDefaultWidget(right_spacer)

        self.addAction(left_group)
        self.addAction(middle_group)
        self.addAction(right_group)

        self.left_group = left_group
        self.middle_group = middle_group
        self.right_group = right_group

        self.action_toggle_sidebar = self._new_action_toggle_records()
        self.insertAction(self.left_group, self.action_toggle_sidebar)
        
        self.action_record = self._new_action_record()
        self.insertAction(self.middle_group, self.action_record)
        
        self.action_toggle_config = self._new_action_toggle_controls()
        self.insertAction(self.right_group, self.action_toggle_config)

    def _new_action_toggle_records(self):
        i = icon(MDI7.dock_left, color=UIColors.ON_PRIMARY)
        a = QAction(i, "Records Explorer", self)
        a.setCheckable(True)
        a.toggled.connect(self.records_panel_toggled.emit)
        a.setShortcut("Ctrl+E")
        return a

    def _new_action_toggle_controls(self):
        i = icon(MDI7.dock_right, color=UIColors.ON_PRIMARY)
        a = QAction(i, "Control Panel", self)
        a.setCheckable(True)
        a.toggled.connect(self.control_panel_toggled.emit)
        a.setShortcut("Ctrl+C")
        return a

    def _new_action_record(self):
        i_rec = icon(MDI7.record_rec, color=UIColors.RED)
        i_end = icon(MDI7.stop_circle_outline, color=UIColors.RED)

        a = QAction(i_rec, "Start Recording", self)
        a.setCheckable(True)
        a.setChecked(False)
        a.setShortcut("Ctrl+R")

        def toggle_record(recording_started: bool):
            if recording_started:
                self._parent.setStyleSheet("QWidget#central { border: 3px solid red; } ")
                a.setIcon(i_end)
                # self.record_toggled.emit(True)
            else:
                self._parent.setStyleSheet("")
                a.setIcon(i_rec)
                # self.record_toggled.emit(False)

        a.triggered.connect(toggle_record)
        return a
