import numpy as np
from typing import Iterable
from time import perf_counter
from enum import StrEnum

from PySide6.QtGui import QFont, QColor, QIcon
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QSizePolicy,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QLabel,
)

from pyqtgraph import PlotWidget, mkPen, InfiniteLine
from PySide6.QtGui import QPixmap, QPainter

from PySide6.QtCore import QDir
from utils.paths import PATH_STATIC
QDir.addSearchPath("icons", PATH_STATIC)

# Theme
# -------------------------------------------------------------------------------------------------


class Fonts:
    OpenGLMonospace = QFont("Monospace", 8)
    GUIMonospace = QFont("Monospace", 10)


class GLColors:
    WHITE = QColor(255, 255, 255, 255)
    BLACK = QColor(0, 0, 0, 255)

    RED = QColor(255, 40, 40, 255)
    GREEN = QColor(40, 255, 40, 255)
    BLUE = QColor(40, 40, 255, 255)

    PURPLE = QColor(136, 97, 170, 255)
    TURQUOIS = QColor(23, 155, 93, 255)
    ORANGE = QColor(255, 165, 0, 255)


class UIColors:
    PRIMARY = "#202020"
    PRIMARY_BUTTON_HOVER = "#272727"
    ON_PRIMARY = "#919090"

    SECONDARY = "#1E1E1E"
    ON_SECONDARY = "#878786"

    FOREGROUND = "#131313"
    ON_FOREGROUND = "#565757"
    ON_FOREGROUND_DIM = "#373737"

    ACCENT = "#1A1A1A"
    ON_ACCENT = "#737473"
    ON_ACCENT_DIM = "#4A4A4A"

    RED = "#d14f3e"
    GREEN = "#3ed14f"
    BLUE = "#3ec0d1"
    ORANGE = "#FFCC99"
    PASTEL_BLUE = "#98f9f9"


# Styles
# -------------------------------------------------------------------------------------------------

MAIN_WINDOW_STYLE = f"""
    QMainWindow {{
        background-color: {UIColors.PRIMARY};
    }}
"""

DOCK_WIDGET_STYLE = f"""
    QDockWidget {{
        background: {UIColors.SECONDARY};
        border: 1px solid {UIColors.ON_FOREGROUND_DIM};
        border-radius: 8px;
    }}
    QDockWidget::title {{
        background: {UIColors.ACCENT};
        color: {UIColors.ON_ACCENT};
        padding: 6px 6px;
        font-weight: bold;
        font-size: 12;
    }}
"""

TOOLTIP_STYLE = f"""
    QToolTip {{
        background-color: {UIColors.FOREGROUND};
        color: {UIColors.ON_FOREGROUND};
        border: 2px solid {UIColors.ON_FOREGROUND_DIM};
        border-radius: 5px;
        white-space: nowrap;
        padding: 2px;
    }}
"""

GROUPBOX_STYLE = f"""
    QGroupBox {{
        border: 2px solid {UIColors.ON_FOREGROUND_DIM};
        border-radius: 5px;
        margin-top: 1ex;
        font-weight: bold;
        padding: 0px;
        padding-top: 3px;
        color: {UIColors.ON_PRIMARY};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left; /* position at the top center */
        left: 10px;
        background-color: {UIColors.PRIMARY}; /* Use primary color for title background */
        color: {UIColors.ON_PRIMARY};
    }}
"""

COMBOBOX_STYLE = f"""
    QComboBox {{
        border: none;
        background-color: transparent;
        padding: 1px 5px 1px 3px;
    }}
    QComboBox:focus {{
        outline: none;
    }}
    QComboBox:hover {{
        background-color: {UIColors.PRIMARY_BUTTON_HOVER};
        border-radius: 5px;
    }}
    QComboBox::drop-down:button {{
        background-color: {UIColors.PRIMARY_BUTTON_HOVER};
        width: 36px;
        border: none;
        border-radius: 5px;
    }}
    QComboBox::down-arrow {{
        image: url(icons:chevron-down.svg);
    }}
    QComboBox QAbstractItemView {{
        background: transparent;
        border: none;
        outline: none;
    }}
    QComboBox QAbstractItemView::item {{
    background-color: {UIColors.PRIMARY};
        height: 40px;
    }}
    QComboBox QAbstractItemView::item:hover {{
        background-color: {UIColors.PRIMARY_BUTTON_HOVER};
        border: none;
    }}
    QComboBox QAbstractItemView:focus {{
        outline: none;
    }}
"""


STATUSBAR_STYLE = f"""
    QStatusBar {{
        background-color: {UIColors.ACCENT};
        color: {UIColors.ON_ACCENT};
    }}
"""

APP_STYLE_LIST = [
    MAIN_WINDOW_STYLE,
    DOCK_WIDGET_STYLE,
    TOOLTIP_STYLE,
    GROUPBOX_STYLE,
    STATUSBAR_STYLE,
]

APP_STYLE = "\n".join(APP_STYLE_LIST)


# Debug
# -------------------------------------------------------------------------------------------------

class QSSDebug:
    """Adds """
    def __init__(self, widget: QWidget):
        self.widget = widget
        self.is_active = False
        self.debug_qss = """* {
            background-color: rgba(0, 155, 0, 25);
            /*border: 1px solid;
            border-color: rgba(0, 255, 0, 25);*/
        }"""
        self.original_qss = ""

    def toggle(self):
        if not self.is_active:
            self.original_qss = self.widget.styleSheet()
            self.widget.setStyleSheet(self.original_qss + self.debug_qss)
            self.is_active = True
        else:
            self.widget.setStyleSheet(self.original_qss)
            self.is_active = False


# Helpers
# -------------------------------------------------------------------------------------------------


def toggle_widget(w: QWidget):
    w.close() if w.isVisible() else w.show()

def svg_icon(name: str):
    pixmap = QPixmap(f"icons:{name}.svg")
    painter = QPainter(pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), UIColors.ON_PRIMARY)
    painter.end()
    return QIcon(pixmap)

# Widgets
# -------------------------------------------------------------------------------------------------


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

    def update(self, pixmap):
        """Set the pixmap and fit it to the view."""
        self.pixmap_item.setPixmap(pixmap)
        if not pixmap.isNull():
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

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
            angle=90, movable=False, pen=mkPen(color=UIColors.ON_ACCENT, width=1)
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


class EMALatencyLabel(QLabel):
    """Exponential Moving Average latency measurement"""

    def __init__(self, name: str, label_update_freq=120):
        self._name = name
        self._label_update_freq = label_update_freq

        super().__init__(f"{self._name}: --")
        self.setStyleSheet(f"color: {UIColors.ON_FOREGROUND};")

        self._last_time = perf_counter()
        self._avg_dt = self._last_time
        self._counter = 0
        self.alpha = 0.2

    def update(self):
        t = perf_counter()
        dt = t - self._last_time
        self._avg_dt = (1 - self.alpha) * self._avg_dt + self.alpha * dt
        if self._counter == 0:
            self.setText(f"{self._name}: {self._avg_dt:.3f} ({int(1/self._avg_dt)})")
        self._last_time = t
        self._counter = (self._counter + 1) % self._label_update_freq
