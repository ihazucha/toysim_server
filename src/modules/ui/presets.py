from PySide6.QtGui import QFont, QColor

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QSizePolicy,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
)

class Fonts:
    Monospace = QFont("Monospace", 8)

class MColors:
    WHITE = QColor(255, 255, 255, 255)
    GRAY = QColor(125, 125, 125, 255)
    GRAY_TRANS = QColor(185, 185, 185, 150)
    RED = QColor(255, 40, 40, 255)
    GREEN = QColor(40, 255, 40, 255)
    BLUE = QColor(40, 40, 255, 255)
    BROWN = QColor(119, 49, 19, 255)
    DARK_BROWN = QColor(101, 67, 33, 255)
    PURPLISH = QColor(93, 23, 155, 255)
    PURPLISH_LIGHT = QColor(136, 97, 170, 255)
    TURQUOIS = QColor(23, 155, 93, 255)
    ORANGE = QColor(255, 165, 0, 255)

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
