from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QToolBar, QWidget, QStyle, QWidgetAction, QSizePolicy
from PySide6.QtGui import QIcon, QAction, QPixmap, QPainter

from utils.paths import icon_path
from enum import Enum


def create_icon_with_white_background(standard_icon, style, size=24):
    """Create an icon with white background from a standard icon."""
    # Get the standard icon as a pixmap
    pixmap = style.standardIcon(standard_icon).pixmap(size, size)

    # Create a new pixmap with white background
    white_pixmap = QPixmap(size, size)
    white_pixmap.fill(Qt.white)

    # Paint the original icon onto the white background
    painter = QPainter(white_pixmap)
    painter.drawPixmap(0, 0, pixmap)
    painter.end()

    # Convert back to QIcon
    return QIcon(white_pixmap)


class DataSource(Enum):
    LIVE = 1
    RECORD = 2


class TopToolBar(QToolBar):
    records_panel_toggled = Signal(bool)
    source_toggled = Signal(DataSource)
    record_toggled = Signal(bool)
    playback_toggled = Signal(bool)
    config_panel_toggled = Signal(bool)

    def __init__(self, parent: QWidget, name="TopToolBar"):
        super().__init__(name)
        self._parent = parent
        self._init_layout()

    def _init_layout(self):
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

        self.action_toggle_sidebar = self._create_action_toggle_sidebar()
        self.insertAction(self.left_group, self.action_toggle_sidebar)
        self.action_toggle_source = self._create_action_toggle_source()
        self.insertAction(self.left_group, self.action_toggle_source)
        
        self.action_record = self._create_action_record()
        self.insertAction(self.middle_group, self.action_record)
        self.action_toggle_playback_play_pause = self._create_action_toggle_playback_play_pause()
        self.insertAction(self.middle_group, self.action_toggle_playback_play_pause)
        
        self.action_toggle_config = self._create_action_toggle_config()
        self.insertAction(self.right_group, self.action_toggle_config)

    def _create_action_toggle_sidebar(self):
        icon = self.style().standardIcon(QStyle.SP_DirIcon)
        a = QAction(icon, "Sidebar", self)
        a.setCheckable(True)
        a.toggled.connect(self.records_panel_toggled.emit)
        a.setShortcut("Ctrl+E")
        return a

    def _create_action_toggle_source(self):
        icon_live = self.style().standardIcon(QStyle.SP_DriveNetIcon)
        icon_record = self.style().standardIcon(QStyle.SP_DriveHDIcon)
        a = QAction(icon_record, "Source", self)
        a.setCheckable(True)
        a.setChecked(False)
        a.setShortcut("Ctrl+S")

        def toggle_live():
            a.setIcon(icon_record)
            self.source_toggled.emit(DataSource.LIVE)

        def toggle_record():
            a.setIcon(icon_live)
            self.source_toggled.emit(DataSource.RECORD)

        def toggle(state: bool):
            toggle_record() if state else toggle_live()

        a.toggled.connect(toggle)
        return a

    def handle_record_selected(self):
        self.action_toggle_source.setChecked(True)
        self.source_toggled.emit(DataSource.RECORD)

    def _create_action_toggle_config(self):
        icon = self.style().standardIcon(QStyle.SP_FileDialogDetailedView)
        a = QAction(icon, "Config", self)
        a.setCheckable(True)
        a.toggled.connect(self.config_panel_toggled.emit)
        a.setShortcut("Ctrl+C")
        return a

    def _create_action_record(self):
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
        return a

    def _create_action_toggle_playback_play_pause(self):
        # TODO: use iconset when finished implementing
        icon_play = create_icon_with_white_background(QStyle.SP_MediaPlay, self.style())
        icon_pause = create_icon_with_white_background(QStyle.SP_MediaPause, self.style())

        a = QAction(icon_play, "Playback", self)
        a.setShortcut("Ctrl+Space")
        a.setCheckable(True)
        a.setChecked(False)
        a.setVisible(False)

        def set_icon_state(state: bool):
            a.setChecked(state)
            a.setIcon(icon_pause if state else icon_play)
        
        # Self triggers
        def toggle_playback(state: bool):
            self.playback_toggled.emit(state)

        a.triggered.connect(toggle_playback)
        self.playback_toggled.connect(set_icon_state)

        # Remote triggers
        def update_on_source_toggled(source: DataSource):
            a.setVisible(source == DataSource.RECORD)
            a.setEnabled(source == DataSource.RECORD)
            a.setChecked(False)
            toggle_playback(False)

        self.source_toggled.connect(update_on_source_toggled)
        return a