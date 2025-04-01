from PySide6.QtGui import QFont

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QSizePolicy,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
)


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
