import numpy as np
from typing import Tuple
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QWidget, QVBoxLayout

from pyqtgraph import PlotWidget, PlotCurveItem, mkPen

from modules.ui.plots import PlotStatsWidget, STEP_TICKS, PLOT_TIME_STEPS
from modules.ui.presets import Colors, TooltipLabel


class IMURawWidget(QWidget):
    def __init__(self, title: str, axis_names=("X", "Y", "Z"), axis_tooltips=("", "", "")):
        super().__init__()
        self.plot_widget = IMURawPlotWidget(title)
        # self.stats_widget = IMURawStatsWidget(axis_names, axis_tooltips)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget, stretch=1)
        # layout.addWidget(self.stats_widget)

    def update(self, xyz: Tuple[float, float, float]):
        self.plot_widget.update(xyz)
        # self.stats_widget.update(xyz[-1])


class IMURawStatsWidget(PlotStatsWidget):
    def __init__(self, axis_names=("X", "Y", "Z"), axis_tooltips=("", "", "")):
        super().__init__()

        html_value = "<span style='color: {}; font-weight: bold; font-family: Courier New, monospace;'>{{}}</span>"

        self._axes = (
            dict(
                id="x",
                fstr=f"{axis_names[0]}: {html_value.format(Colors.RED)}",
                label=None,
            ),
            dict(
                id="y",
                fstr=f"{axis_names[1]}: {html_value.format(Colors.GREEN)}",
                label=None,
            ),
            dict(
                id="z",
                fstr=f"{axis_names[2]}: {html_value.format(Colors.BLUE)}",
                label=None,
            ),
        )

        for i, a in enumerate(self._axes):
            a["label"] = TooltipLabel(text=a["fstr"].format("--"), tooltip=axis_tooltips[i])
            self.layout.addWidget(a["label"])

        self.layout.addStretch(1)
 
    def update(self, gyro: Tuple[float, float, float]):
        for i, a in enumerate(self._axes):
            str = "{:6.2f}".format(gyro[i]).replace(" ", "&nbsp;")
            a["label"].setText(a["fstr"].format(str))


class IMURawPlotWidget(PlotWidget):
    def __init__(self, title, axis_names=("x", "y", "z")):
        super().__init__()
        self.setBackground(Colors.FOREGROUND)
        self.getPlotItem().setTitle(title)
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
        self._axes = (
            dict(id="x", name=axis_names[0], color=Colors.RED, curve=None),
            dict(id="y", name=axis_names[1], color=Colors.GREEN, curve=None),
            dict(id="z", name=axis_names[2], color=Colors.BLUE, curve=None),
        )

        self._n_steps = np.array(PLOT_TIME_STEPS)

        self._rescale_frequency = 15
        self._rescale_counter = 0
        self._rescale_min_range = 2.0
        self._rescale_y_padding = 0.5

        self._setup_axes()
        self._setup_legend()
        self._setup_plots()

    def _setup_axes(self):
        self.setYRange(-2.0, 2.0, padding=0)

        pen = mkPen(Colors.ON_ACCENT)

        self.getAxis("left").setPen(pen)
        self.getAxis("left").setTextPen(pen)

        self.getAxis("bottom").setTicks(STEP_TICKS)
        self.getAxis("bottom").setPen(pen)
        self.getAxis("bottom").setTextPen(pen)

    def _setup_legend(self):
        self.legend = self.getPlotItem().addLegend()
        self.legend.anchor(itemPos=(0, 0), parentPos=(0, 1), offset=(15, -35))
        self.legend.setBrush(QBrush(QColor(Colors.ACCENT)))
        self.legend.setPen(mkPen(color=Colors.ON_FOREGROUND, width=0.5))
        self.legend.layout.setContentsMargins(3, 1, 3, 1)
        self.legend.setColumnCount(3)

    def _setup_plots(self):
        for ad in self._axes:
            pen = mkPen(ad["color"], style=Qt.PenStyle.SolidLine, width=1)
            ad["curve"] = PlotCurveItem(name=ad["name"], pen=pen, antialias=True, skipFiniteCheck=True)
            self.addItem(ad["curve"])
            # self.legend.addItem(ad["curve"], name=ad["name"])

    def update(self, axes_data: np.ndarray):
        for i in range(len(self._axes)):
            self._axes[i]["curve"].setData(self._n_steps, axes_data[:, i])

        # if self._rescale_counter == 0:
            # self._update_y_scale()
        # self._rescale_counter = (self._rescale_counter + 1) % self._rescale_frequency



    def _update_y_scale(self):
        data_min = np.min(self._target_speed_data)
        data_max = np.max(self._target_speed_data)

        range_min = data_min - self._rescale_y_padding
        range_max = data_max + self._rescale_y_padding

        if range_max - range_min < self._rescale_min_range:
            mid = (range_max + range_min) / 2
            range_min = mid - self._rescale_min_range / 2
            range_max = mid + self._rescale_min_range / 2

        # Only update if change diff > 20%
        current_min, current_max = self.getViewBox().viewRange()[1]
        min_changed = abs(current_min - range_min) > abs(current_min * 0.2)
        max_changed = abs(current_max - range_max) > abs(current_max * 0.2)

        if min_changed or max_changed:
            self.setYRange(range_min, range_max, padding=0)
