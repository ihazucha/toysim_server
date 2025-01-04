import os

from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QListWidget

from utils.data import PATH_RECORDS


class RecordSidebar(QDockWidget):
    def __init__(self, parent=None):
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

    def load_records(self):
        records = os.listdir(PATH_RECORDS)
        for record in records:
            self.record_list.addItem(record)

    def add_record(self, record_name):
        self.record_list.addItem(record_name)
