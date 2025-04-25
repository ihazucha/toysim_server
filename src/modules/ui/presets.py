import numpy as np
from typing import Iterable
from time import perf_counter

from PySide6.QtGui import QFont, QColor
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QSizePolicy,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QLabel,
)

from pyqtgraph import PlotWidget, mkPen, InfiniteLine

class Fonts:
    OpenGLMonospace = QFont("Monospace", 8)
    GUIMonospace = QFont("Monospace", 10)

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
    RED = "#FB9898"
    ORANGE = "#FFCC99"
    PASTEL_BLUE = "#98f9f9"
    PASTEL_PURPLE = "#DDA0DD"
    PASTEL_YELLOW = "#FFFFE0"
    PASTEL_ORANGE = "#f99898"


TOOLTIP_STYLE = f"""
    QToolTip {{
        background-color: {Colors.FOREGROUND};
        color: {Colors.ON_FOREGROUND};
        border: 2px solid {Colors.ON_FOREGROUND_DIM};
        border-radius: 5px;
        white-space: nowrap;
        padding: 2px;
    }}
"""

GROUPBOX_STYLE = f"""
    QGroupBox {{
        border: 2px solid {Colors.ON_FOREGROUND_DIM};
        border-radius: 5px;
        margin-top: 1ex;
        font-weight: bold;
        padding: 0px;
        padding-top: 3px;
        color: {Colors.ON_PRIMARY};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left; /* position at the top center */
        left: 10px;
        background-color: {Colors.PRIMARY}; /* Use primary color for title background */
        color: {Colors.ON_PRIMARY};
    }}
"""

APP_STYLE_LIST = [
    TOOLTIP_STYLE,
    GROUPBOX_STYLE,
]

APP_STYLE = "\n".join(APP_STYLE_LIST)

class TooltipLabel(QLabel):
    def __init__(self, text: str, tooltip: str | None = None, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        if tooltip:
            self.setToolTip(tooltip)
            self.setToolTipDuration(0)

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
        # if not pixmap.isNull():
            # self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        # """Scale the pixmap when the view is resized."""
        if not self.pixmap_item.pixmap().isNull():
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        super().resizeEvent(event)

class PlotWidgetHorizontalCursor:
    def __init__(
        self, plot_widget: PlotWidget, x_values: Iterable, update_values_at_index_callback: callable
    ):
        self.plot_widget = plot_widget
        self.x_values = x_values
        self.update_values_at_index_callback = update_values_at_index_callback

        self.plot_widget.setMouseEnabled(x=True, y=False)

        self.vLine = InfiniteLine(
            angle=90, movable=False, pen=mkPen(color=Colors.ON_ACCENT, width=1)
        )
        self.vLine.setVisible(False)
        self.plot_widget.addItem(self.vLine)

        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def _on_mouse_moved(self, pos):
        if self.plot_widget.getViewBox().sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()
            x = np.clip(x, self.x_values[0], self.x_values[-1])
            self.vLine.setPos(np.round(x))
            self.vLine.setVisible(True)

            nearest_idx = np.abs(self.x_values - x).argmin()
            self.update_values_at_index_callback(nearest_idx)
        else:
            self.vLine.setVisible(False)
            self.update_values_at_index_callback(-1)

class FrameCounter(QObject):
    sigFpsUpdate = Signal(object)

    def __init__(self, interval=1000):
        super().__init__()
        self.count = 0
        self.last_update = 0
        self.interval = interval

    def update(self):
        self.count += 1

        if self.last_update == 0:
            self.last_update = perf_counter()
            self.startTimer(self.interval)

    def timerEvent(self, evt):
        now = perf_counter()
        elapsed = now - self.last_update
        fps = self.count / elapsed
        self.last_update = now
        self.count = 0
        self.sigFpsUpdate.emit(fps)