from PySide6.QtGui import QFont

class DefaultMonospaceFont(QFont):
    def __init__(self, size=10):
        super().__init__()
        self.setStyleHint(QFont.Monospace)
        self.setPointSize(size)