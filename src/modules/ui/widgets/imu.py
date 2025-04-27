import numpy as np
from typing import Tuple
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication

from pyqtgraph import PlotWidget, PlotCurveItem, mkPen
import pyqtgraph as pg

from modules.ui.plots import PlotStatsWidget, STEP_TICKS, PLOT_TIME_STEPS
from modules.ui.presets import UIColors, TooltipLabel


class IMURawWidget(QWidget):
    def __init__(self, title: str, axis_names=("X", "Y", "Z"), axis_tooltips=("", "", "")):
        super().__init__()
        self.plot_widget = IMURawRemotePlotWidget(title)
        self.stats_widget = IMURawStatsWidget(axis_names, axis_tooltips)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget, stretch=1)
        layout.addWidget(self.stats_widget)

    def update(self, xyz: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        self.plot_widget.update_data(xyz)
        self.stats_widget.update(xyz[:, -1])


class IMURawStatsWidget(PlotStatsWidget):
    def __init__(self, axis_names=("X", "Y", "Z"), axis_tooltips=("", "", "")):
        super().__init__()

        html_value = "<span style='color: {}; font-weight: bold; font-family: Courier New, monospace;'>{{}}</span>"

        self._axes = (
            dict(
                id="x",
                fstr=f"{axis_names[0]}: {html_value.format(UIColors.RED)}",
                label=None,
            ),
            dict(
                id="y",
                fstr=f"{axis_names[1]}: {html_value.format(UIColors.GREEN)}",
                label=None,
            ),
            dict(
                id="z",
                fstr=f"{axis_names[2]}: {html_value.format(UIColors.BLUE)}",
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
        self.setBackground(UIColors.FOREGROUND)
        self.getPlotItem().setTitle(title)
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
        self._axes = (
            dict(id="x", name=axis_names[0], color=UIColors.RED, curve=None),
            dict(id="y", name=axis_names[1], color=UIColors.GREEN, curve=None),
            dict(id="z", name=axis_names[2], color=UIColors.BLUE, curve=None),
        )

        self._n_steps = np.array(PLOT_TIME_STEPS)

        self._setup_axes()
        self._setup_legend()
        self._setup_plots()

    def _setup_axes(self):
        self.setYRange(-2.0, 2.0, padding=0)

        pen = mkPen(UIColors.ON_ACCENT)

        self.getAxis("left").setPen(pen)
        self.getAxis("left").setTextPen(pen)

        self.getAxis("bottom").setTicks(STEP_TICKS)
        self.getAxis("bottom").setPen(pen)
        self.getAxis("bottom").setTextPen(pen)

    def _setup_legend(self):
        self.legend = self.getPlotItem().addLegend()
        self.legend.anchor(itemPos=(0, 0), parentPos=(0, 1), offset=(15, -35))
        self.legend.setBrush(QBrush(QColor(UIColors.ACCENT)))
        self.legend.setPen(mkPen(color=UIColors.ON_FOREGROUND, width=0.5))
        self.legend.layout.setContentsMargins(3, 1, 3, 1)
        self.legend.setColumnCount(3)

    def _setup_plots(self):
        for ad in self._axes:
            pen = mkPen(ad["color"], style=Qt.PenStyle.SolidLine, width=1)
            ad["curve"] = PlotCurveItem(
                name=ad["name"], pen=pen, antialias=True, skipFiniteCheck=True
            )
            self.addItem(ad["curve"])

    def update_data(self, axes_data: np.ndarray):
        for i in range(len(self._axes)):
            self._axes[i]["curve"].setData(self._n_steps, axes_data[:, i])


class IMURawRemotePlotWidget(pg.RemoteGraphicsView):
    def __init__(self, title):
        super().__init__(debug=False)
        self.pg.setConfigOptions(antialias=True)
        self._view.setBackground(UIColors.FOREGROUND)

        self._plt = self.pg.PlotItem()
        self._plt.setTitle(title)
        self._plt.showGrid(x=True, y=True, alpha=0.3)
        self._plt._setProxyOptions(deferGetattr=True)

        self.setCentralItem(self._plt)
        QApplication.instance().aboutToQuit.connect(self.close)

        self._axes_pens = (
            dict(color=UIColors.RED, style=Qt.PenStyle.SolidLine, width=1),
            dict(color=UIColors.GREEN, style=Qt.PenStyle.SolidLine, width=1),
            dict(color=UIColors.BLUE, style=Qt.PenStyle.SolidLine, width=1),
        )

        self._n_steps = np.array(PLOT_TIME_STEPS)

        # Add counters for autoscaling logic
        self._autoscale_counter = 0
        self._autoscale_interval = 30
        self._autoscale_min_y_range = 4.0

        self._setup_axes()

    def _setup_axes(self):
        self._plt.setYRange(-self._autoscale_min_y_range / 2, self._autoscale_min_y_range / 2)

        pen_opts = dict(color=UIColors.ON_ACCENT)

        self._plt.getAxis("left").setPen(pen_opts)
        self._plt.getAxis("left").setTextPen(pen_opts)

        self._plt.getAxis("bottom").setTicks(STEP_TICKS)
        self._plt.getAxis("bottom").setPen(pen_opts)
        self._plt.getAxis("bottom").setTextPen(pen_opts)

    def update_data(self, axes_data: np.ndarray):
        try:
            self._plt.multiDataPlot(
                x=self._n_steps,
                y=axes_data,
                pen=self._axes_pens,
                clear=(True, False, False),
                _callSync="off",
            )
            self._plt.scatterPlot(
                x=[self._n_steps[-1]]*3,
                y=axes_data[:, -1],
                pen=None,
                brush=[ax["color"] for ax in self._axes_pens],
                size=5,
                # clear=(True, False, False),
                _callSync="off",
            )
            self._autoscale_counter += 1
            self._autoscale(axes_data)
        except pg.multiprocess.remoteproxy.ClosedError:
            pass

    def _autoscale(self, axes_data):
        # Check if we should autoscale
        if self._autoscale_counter >= self._autoscale_interval:
            self._autoscale_counter = 0

            # Get data min/max
            data_min = axes_data.min()
            data_max = axes_data.max()

            # Calculate range with padding
            data_range = data_max - data_min
            padding = data_range * 0.1
            y_min = data_min - padding
            y_max = data_max + padding

            # Ensure minimum range is preserved
            if (y_max - y_min) < self._autoscale_min_y_range:
                center = (y_max + y_min) / 2
                y_min = center - self._autoscale_min_y_range / 2
                y_max = center + self._autoscale_min_y_range / 2

            # Update Y range
            self._plt.setYRange(y_min, y_max, _callSync="off")
