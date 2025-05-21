from PySide6.QtCore import Signal, Slot, QSize
from PySide6.QtWidgets import QToolBar, QWidget, QWidgetAction, QSizePolicy
from PySide6.QtGui import QAction

from modules.ui.presets import UIColors, svg_icon

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

    def _init_layout(self):
        self.setIconSize(QSize(36, 36))
        self.setMovable(False)
        self.setStyleSheet(
            f"""
            QToolBar {{
                padding-left: 6px;
                padding-right: 6px;
                border: none;
            }}
            QToolBar QToolButton {{
                background-color: transparent;
                border-radius: 5px;
            }}
            QToolBar QToolButton:hover {{
                background-color: {UIColors.PRIMARY_BUTTON_HOVER};
            }}
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
        i_closed = svg_icon("dock-left")
        i_opened = svg_icon("dock-left-open")
        a = QAction(i_closed, "Records Explorer", self)
        a.setCheckable(True)

        @Slot(bool)
        def toggle_records(checked: bool):
            a.setIcon(i_opened if checked else i_closed)        
            self.records_panel_toggled.emit(checked)

        a.toggled.connect(toggle_records)
        a.setShortcut("Ctrl+[")
        return a

    def _new_action_toggle_controls(self):
        i_closed = svg_icon("dock-right")
        i_opened = svg_icon("dock-right-open")
        a = QAction(i_closed, "Control Panel", self)
        a.setCheckable(True)

        @Slot(bool)
        def toggle_controls(opened: bool):
            a.setIcon(i_opened if opened else i_closed)
            self.control_panel_toggled.emit(opened)
        
        a.toggled.connect(toggle_controls)
        a.setShortcut("Ctrl+]")
        return a

    def _new_action_record(self):
        i_rec = icon(MDI7.record_circle_outline, color=UIColors.RED)
        i_end = icon(MDI7.stop_circle_outline, color=UIColors.ON_PRIMARY)

        a = QAction(i_rec, "Start Recording", self)
        a.setCheckable(True)
        a.setChecked(False)
        a.setShortcut("Ctrl+R")

        def toggle_record(recording_started: bool):
            if recording_started:
                self._parent.setStyleSheet("QWidget#central { border: 3px solid red; } ")
                a.setIcon(i_end)
                self.record_toggled.emit(True)
            else:
                self._parent.setStyleSheet("")
                a.setIcon(i_rec)
                self.record_toggled.emit(False)

        a.triggered.connect(toggle_record)
        return a
