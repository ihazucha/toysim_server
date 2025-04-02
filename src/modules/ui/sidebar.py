import os
from enum import IntEnum

from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QMenu, QApplication
from PySide6.QtCore import Signal, Qt, QPoint
from PySide6.QtGui import QAction

from utils.paths import PATH_RECORDS
from datetime import datetime

class CustomListItemRoles(IntEnum):
    ID = Qt.ItemDataRole.UserRole

class RecordsSidebar(QDockWidget):
    record_selected = Signal(str)

    def __init__(self, parent=None, default_closed=True):
        super().__init__(parent)
        self.setWindowTitle("Record Sidebar")
        self.setFixedWidth(200)
        widget = QWidget()
        self.layout = QVBoxLayout(widget)
        widget.setLayout(self.layout)
        
        self.record_list = QListWidget(widget)
        self.record_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.record_list.setStyleSheet("""
            QListWidget::item:selected {
                background-color: #3daee9;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #e5f3ff;
            }
        """)
        
        self.record_list.itemDoubleClicked.connect(self._on_record_selected)

        self.record_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.record_list.customContextMenuRequested.connect(self._show_context_menu)

        self.layout.addWidget(self.record_list)
        self.load_records()
        self.setWidget(widget)

        if default_closed:
            self.close()

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
        display_name = f"{datetime.fromtimestamp(int(timestamp) / 1e9).strftime('%d-%m-%Y %H:%M:%S')}"
        
        item = QListWidgetItem(display_name)
        item.setData(CustomListItemRoles.ID, timestamp)
        self.record_list.addItem(item)
    
    def _on_record_selected(self, item: QListWidgetItem):
        timestamp = item.data(CustomListItemRoles.ID)
        self.record_selected.emit(timestamp)

    
    def _show_context_menu(self, position: QPoint):
        """Show context menu for list items."""
        item = self.record_list.itemAt(position)
        if not item:
            return
            
        # Create menu
        context_menu = QMenu(self)
        
        # Add Copy Timestamp action
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