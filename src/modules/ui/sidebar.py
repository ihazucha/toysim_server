import os

from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QListWidget

from utils.paths import PATH_RECORDS
from datetime import datetime

class RecordsSidebar(QDockWidget):
    def __init__(self, parent=None, default_closed=True):
        super().__init__(parent)
        self.setWindowTitle("Record Sidebar")
        self.setFixedWidth(200)
        widget = QWidget()
        self.layout = QVBoxLayout(widget)
        widget.setLayout(self.layout)
        self.record_list = QListWidget(widget)
        self.layout.addWidget(self.record_list)
        self.load_records()
        self.setWidget(widget)
        if default_closed:
            self.close()

    def load_records(self):
        records = os.listdir(PATH_RECORDS)
        for record in records:
            try:
                timestamp = int(record.split(".")[0])
                record = f"{datetime.fromtimestamp(timestamp / 1e9).strftime('%d-%m-%Y %H:%M:%S')}"
                self.record_list.addItem(record)
            except Exception as e:
                print(f"[{self.__class__.__name__}] Encountered invalid record name format: {record}")
                print(e)

    def add_record(self, record_name):
        self.record_list.addItem(record_name)
