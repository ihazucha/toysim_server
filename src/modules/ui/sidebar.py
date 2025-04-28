import os
from enum import IntEnum
from datetime import datetime


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

from utils.paths import PATH_RECORDS
from modules.ui.presets import UIColors

class CustomListItemRoles(IntEnum):
    ID = Qt.ItemDataRole.UserRole

class RecordItemWidget(QWidget):
    """Custom widget for displaying record items with preview image and date."""
    
    def __init__(self, timestamp, preview_image_path=None, parent=None):
        super().__init__(parent)
        self.timestamp = timestamp
        
        # Convert timestamp to datetime
        dt = datetime.fromtimestamp(int(timestamp) / 1e9)
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create image label
        self.image_label = QLabel()
        self.image_label.setFixedSize(64, 64)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(f"""
            background-color: {UIColors.FOREGROUND};
            border-radius: 5px;
            /* border: 1px solid {UIColors.ON_FOREGROUND_DIM}; */
            font-style: italic;
            color: {UIColors.ON_FOREGROUND}
        """)
        
        # Set image if provided, otherwise use placeholder
        if preview_image_path and os.path.exists(preview_image_path):
            pixmap = QPixmap(preview_image_path)
            pixmap = pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText("No\nPreview")
            self.image_label.setAlignment(Qt.AlignCenter)
            
        # Create date/time layout and labels
        date_layout = QVBoxLayout()
        date_layout.setSpacing(2)
        
        # Date label (larger, bold)
        self.date_label = QLabel(dt.strftime("%d %b %Y"))
        self.date_label.setStyleSheet(f"""
            font-weight: bold;
            font-size: 12px;
        """)
        
        # Time label (smaller)
        self.time_label = QLabel(dt.strftime("%H:%M:%S"))
        self.time_label.setStyleSheet("""
            font-size: 10px;
        """)
        
        date_layout.addWidget(self.date_label)
        date_layout.addWidget(self.time_label)
        date_layout.addStretch()
        
        # Add widgets to main layout
        layout.addWidget(self.image_label)
        layout.addLayout(date_layout, 1)
        
        self.setLayout(layout)
        
    def sizeHint(self):
        return QSize(180, 82)  # 64px height + 10px margins


class RecordsSidebar(QDockWidget):
    record_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Records")
        # self.setFixedWidth(200)

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
                padding: 5px;
            }}
            QListWidget::item {{
                background-color: {UIColors.ACCENT};
                color: red;
                border: 1px solid {UIColors.ON_FOREGROUND_DIM};
                border-radius: 5px;
                margin-bottom: 5px;
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

        records = os.listdir(PATH_RECORDS)
        records.sort(key=lambda x: int(x.split(".")[0]), reverse=True)
        for record in records:
            try:
                self.add_record(record)
            except Exception as e:
                print(f"[{self.__class__.__name__}] Invalid record name format: {record}")
                print(e)

    def add_record(self, record_name):
        timestamp = record_name.split(".")[0]
        
        # Check if preview image exists (optional)
        preview_path = os.path.join(PATH_RECORDS, f"{timestamp}_preview.png")
        if not os.path.exists(preview_path):
            preview_path = None
        
        # Create custom item widget
        item_widget = RecordItemWidget(timestamp, preview_path)
        
        # Create list item and set size
        list_item = QListWidgetItem(self.record_list)
        list_item.setData(CustomListItemRoles.ID, timestamp)
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
