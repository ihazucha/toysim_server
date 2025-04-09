from PySide6.QtGui import QFont, QColor

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QSizePolicy,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
)

class MColors:
    WHITE = QColor(255, 255, 255, 255)
    GRAY = (125, 125, 125, 255)
    RED = QColor(255, 40, 40, 255)
    GREEN = QColor(40, 255, 40, 255)
    BLUE = QColor(40, 40, 255, 255)
    BROWN = QColor(119, 49, 19, 255)
    DARK_BROWN = QColor(101, 67, 33, 255)
    BROWNISH = QColor("#6d593d")
    GREENISH = QColor("#3d6d59")
    PURPLISH = QColor("#77332d")

class Colors:
    PRIMARY = "#202020"
    ON_PRIMARY = "#919090"
    SECONDARY = "#1E1E1E"
    ON_SECONDARY = "#878786"
    FOREGROUND = "#131313"
    ON_FOREGROUND = "#565757"
    ON_FOREGROUND_DIM = "#373737"
    ACCENT = "#1A1A1A"
    ON_ACCENT = "#737473"
    ON_ACCENT_DIM = "#4A4A4A"
    GREEN = "#98FB98"
    ORANGE = "#FFCC99"
    PASTEL_BLUE = "#ADD8E6"
    PASTEL_PURPLE = "#DDA0DD"
    PASTEL_YELLOW = "#FFFFE0"

class DefaultMonospaceFont(QFont):
    def __init__(self, size=10):
        super().__init__()
        self.setStyleHint(QFont.Monospace)
        self.setPointSize(size)

class FitGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)

        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_pixmap(self, pixmap):
        """Set the pixmap and fit it to the view."""
        self.pixmap_item.setPixmap(pixmap)
        if not pixmap.isNull():
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        # """Scale the pixmap when the view is resized."""
        if not self.pixmap_item.pixmap().isNull():
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        super().resizeEvent(event)
