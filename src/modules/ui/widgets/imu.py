import numpy as np
from typing import Tuple
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication

from pyqtgraph import PlotWidget, PlotCurveItem, mkPen
import pyqtgraph as pg

from modules.ui.plots import PlotStatsWidget, STEP_TICKS, PLOT_TIME_STEPS
from modules.ui.presets import Colors, TooltipLabel


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

    def update(self, xyz: Tuple[float, float, float]):
        self.plot_widget.update_data(xyz)
        self.stats_widget.update(xyz[-1])


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
            ad["curve"] = PlotCurveItem(
                name=ad["name"], pen=pen, antialias=True, skipFiniteCheck=True
            )
            self.addItem(ad["curve"])

    def update_data(self, axes_data: np.ndarray):
        for i in range(len(self._axes)):
            self._axes[i]["curve"].setData(self._n_steps, axes_data[:, i])


class IMURawRemotePlotWidget(pg.RemoteGraphicsView):
    def __init__(self, title, axis_names=("x", "y", "z")):
        super().__init__(debug=False)
        self.pg.setConfigOptions(antialias=True)
        self._plt = self.pg.PlotItem()
        self._view.setBackground(Colors.FOREGROUND)
        self._plt.setTitle(title)
        self._plt.showGrid(x=True, y=True, alpha=0.3)
        self._plt._setProxyOptions(deferGetattr=True)

        self.setCentralItem(self._plt)
        QApplication.instance().aboutToQuit.connect(self.close)

        self._axes = (
            dict(id="x", name=axis_names[0], color=Colors.RED, curve=None),
            dict(id="y", name=axis_names[1], color=Colors.GREEN, curve=None),
            dict(id="z", name=axis_names[2], color=Colors.BLUE, curve=None),
        )

        self._n_steps = np.array(PLOT_TIME_STEPS)

        self._setup_axes()
        # self._setup_legend()
        self._setup_curves()

    def _setup_axes(self):
        self._plt.setYRange(-2.0, 2.0, padding=0)

        pen_opts = dict(color=Colors.ON_ACCENT)

        self._plt.getAxis("left").setPen(pen_opts)
        self._plt.getAxis("left").setTextPen(pen_opts)

        self._plt.getAxis("bottom").setTicks(STEP_TICKS)
        self._plt.getAxis("bottom").setPen(pen_opts)
        self._plt.getAxis("bottom").setTextPen(pen_opts)

    #
    def _setup_legend(self):
        self.legend = self._plt.addLegend()
        self.legend.anchor(itemPos=(0, 0), parentPos=(0, 1), offset=(15, -35))
        self.legend.setBrush(Colors.ACCENT)
        self.legend.setPen(dict(color=Colors.ON_FOREGROUND, width=0.5))
        self.legend.layout.setContentsMargins(3, 1, 3, 1)
        self.legend.setColumnCount(3)

    def _setup_curves(self):
        for ax in self._axes:
            pen = dict(color=ax["color"], style=Qt.PenStyle.SolidLine, width=1)
            ax["curve"] = self._plt.plot(name=ax["id"].capitalize(), pen=pen)

    def update_data(self, axes_data: np.ndarray):
        try:
            for i, ax in enumerate(self._axes):
                pen = dict(color=ax["color"], style=Qt.PenStyle.SolidLine, width=1)
                self._plt.plot(self._n_steps, axes_data[:, i], pen=pen, clear=i==0, _callSync="off")
        except pg.multiprocess.remoteproxy.ClosedError:
            pass
