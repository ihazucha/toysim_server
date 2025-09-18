import os
import traceback
import re

from enum import IntEnum
from datetime import datetime
from pickle import UnpicklingError

from PySide6.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QApplication,
    QLabel,
)
from PySide6.QtCore import Signal, Qt, QPoint, QSize
from PySide6.QtGui import QAction, QPixmap

from modules.ui.presets import UIColors
from modules.recorder import RecordReader
from modules.ui.data import jpg2qimg, rgb2qimg
from datalink.data import ProcessedRealData, ProcessedSimData
from utils.paths import PATH_RECORDS, record_path
from utils.env import DEBUG, pwarn, perror


class CustomListItemRoles(IntEnum):
    ID = Qt.ItemDataRole.UserRole


class RecordItemWidget(QWidget):
    """Custom widget for displaying record items with preview image and date."""

    def __init__(self, date_str: str, time_str: str, thumbnail: QPixmap = None, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.image_label = QLabel()
        self.image_label.setFixedSize(64, 64)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            f"""
            background-color: {UIColors.FOREGROUND};
            border-radius: 5px;
            /* border: 1px solid {UIColors.ON_FOREGROUND_DIM}; */
            font-style: italic;
            color: {UIColors.ON_FOREGROUND}
        """
        )

        if thumbnail is not None:
            self.image_label.setPixmap(thumbnail)
        else:
            self.image_label.setText("No\nPreview")
            self.image_label.setAlignment(Qt.AlignCenter)

        date_layout = QVBoxLayout()
        date_layout.setSpacing(2)

        self.date_label = QLabel(date_str)
        self.date_label.setStyleSheet(
            f"""
            font-weight: bold;
            font-size: 12px;
        """
        )

        self.time_label = QLabel(time_str)
        self.time_label.setStyleSheet(
            """
            font-size: 10px;
        """
        )

        date_layout.addWidget(self.date_label)
        date_layout.addWidget(self.time_label)
        date_layout.addStretch()

        layout.addWidget(self.image_label)
        layout.addLayout(date_layout, 1)

        self.setLayout(layout)

    def sizeHint(self):
        return QSize(160, 82)


class RecordsSidebar(QDockWidget):
    record_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Records")

        widget = QWidget()
        self.layout = QVBoxLayout(widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(self.layout)

        self.record_list = QListWidget(widget)
        self.record_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.record_list.setFrameShape(QListWidget.NoFrame)
        self.record_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.record_list.itemDoubleClicked.connect(self._on_record_selected)
        self.record_list.customContextMenuRequested.connect(self._show_context_menu)
        self.record_list.setStyleSheet(
            f"""
            QListWidget {{
                background-color: {UIColors.SECONDARY};
                border: none;
                padding-left: 10px;
                padding-top: 10px;
                padding-bottom: 10px;
                padding-right: 0px;
                margin-left: 0px;

            }}
            QListWidget::item {{
                background-color: {UIColors.ACCENT};
                border: 1px solid {UIColors.ON_FOREGROUND_DIM};
                border-radius: 5px;
                margin-bottom: 8px;
                margin-right: 5px;
            }}
            QListWidget::item:selected {{
                background-color: {UIColors.FOREGROUND};
                color: {UIColors.ORANGE};
                border: 2px solid {UIColors.ON_FOREGROUND};
            }}
            QListWidget::item:hover {{
                background-color: {UIColors.FOREGROUND};
            }}
            QListWidget::item:selected:active {{
                border: 2px solid {UIColors.ON_SECONDARY};
            }}
            QListWidget::item:selected:!active {{
                border: 2px solid {UIColors.ON_FOREGROUND};
            }}
            QListWidget::focus {{
                outline: none;
            }}
        """
        )

        self.layout.addWidget(self.record_list)
        self.load_records()
        self.setWidget(widget)
        # self.close()

    def load_records(self):
        self.record_list.clear()

        if not os.path.exists(PATH_RECORDS):
            pwarn(f"[{self.__class__.__name__}] Records directory does not exist: {PATH_RECORDS}")
            return

        records = os.listdir(PATH_RECORDS)
        records.sort(key=lambda x: int(x.split(".")[0]), reverse=True)
        for record in records:
            try:
                if not re.match(r"^\d+\.pickle$", record):
                    pwarn(f"[{self.__class__.__name__}] Invalid record name format: {record}")
                    continue
                timestamp = record.split(".")[0]
                self.add_record(timestamp)
            except Exception as e:
                perror(f"[{self.__class__.__name__}] Unable to read record: {record} | Error: {e}")
                if DEBUG == 2:
                    traceback.print_exc()

    def add_record(self, record_timestamp: str):
        data = None
        for data_cls in [ProcessedRealData, ProcessedSimData]:
            try:
                data = RecordReader.read_one(record_path(record_timestamp), ProcessedRealData)
            except UnpicklingError as e:
                pwarn(
                    f"[{self.__class__.__name__}] Unable to unpickle record: {record_timestamp} | Error: {e}"
                )
                if DEBUG == 2:
                    traceback.print_exc()
            if data is not None:
                break

        if data is None:
            return

        qimg = None
        if type(data) == ProcessedRealData:
            qimg = jpg2qimg(data.original.sensor_fusion.camera.jpg)
        elif type(data) == ProcessedSimData:
            qimg = rgb2qimg(data.original.camera.rgb_image)

        dt = datetime.fromtimestamp(int(record_timestamp) / 1e9)
        date_str = dt.strftime("%d %b %Y")
        time_str = dt.strftime("%H:%M:%S")

        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        item_widget = RecordItemWidget(date_str=date_str, time_str=time_str, thumbnail=pixmap)

        # Create list item and set size
        list_item = QListWidgetItem(self.record_list)
        list_item.setData(CustomListItemRoles.ID, record_timestamp)
        list_item.setSizeHint(item_widget.sizeHint())

        # Add widget to list
        self.record_list.addItem(list_item)
        self.record_list.setItemWidget(list_item, item_widget)

    def _on_record_selected(self, item: QListWidgetItem):
        timestamp = item.data(CustomListItemRoles.ID)
        self.record_selected.emit(timestamp)

    def _show_context_menu(self, position: QPoint):
        """Show context menu for list items."""
        item = self.record_list.itemAt(position)
        if not item:
            return

        context_menu = QMenu(self)

        copy_action = QAction("Copy Timestamp", self)
        copy_action.triggered.connect(lambda: self._copy_timestamp(item))
        context_menu.addAction(copy_action)

        open_location_action = QAction("Open File Location", self)
        open_location_action.triggered.connect(lambda: os.startfile(os.path.dirname(record_path(item.data(CustomListItemRoles.ID)))))
        context_menu.addAction(open_location_action)

        context_menu.addSeparator() # Add a separator for better visual grouping

        reload_action = QAction("Reload Records", self)
        reload_action.triggered.connect(self.load_records)
        context_menu.addAction(reload_action)

        # Show menu at cursor position
        global_pos = self.record_list.mapToGlobal(position)
        context_menu.exec_(global_pos)

    def _copy_timestamp(self, item: QListWidgetItem):
        """Copy the timestamp to clipboard."""
        timestamp = item.data(CustomListItemRoles.ID)
        if timestamp:
            clipboard = QApplication.clipboard()
            clipboard.setText(timestamp)
            print(f"Copied timestamp: {timestamp}")
