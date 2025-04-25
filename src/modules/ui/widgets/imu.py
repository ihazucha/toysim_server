import numpy as np
from typing import Tuple
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QWidget, QVBoxLayout

from pyqtgraph import PlotWidget, mkPen

from modules.ui.plots import PlotStatsWidget, DATA_QUEUE_SIZE, STEP_TICKS, PLOT_TIME_STEPS
from modules.ui.presets import Colors, TooltipLabel


class GyroWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.plot_widget = GyroPlotWidget()
        self.stats_widget = GyroPlotStatsWidget()
  
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget, stretch=1)
        layout.addWidget(self.stats_widget)

    def update(self, gyro: Tuple[float, float, float]):
        self.plot_widget.update(gyro)
        self.stats_widget.update(gyro)


class GyroPlotStatsWidget(PlotStatsWidget):
    def __init__(self):
        super().__init__()

        html_colored_number = "<span style='color: {}; font-weight: bold; font-family: Courier New, monospace;'>{{}}</span>"

        self.texts = {
            "x": f"X: {html_colored_number.format(Colors.RED)}",
            "y": f"Y: {html_colored_number.format(Colors.GREEN)}",
            "z": f"Z: {html_colored_number.format(Colors.PASTEL_BLUE)}",
        }

        self.x_label = TooltipLabel(
            text=self.texts["x"].format("--"),
            tooltip="Roll angular velocity",
        )
        self.y_label = TooltipLabel(
            text=self.texts["y"].format("--"),
            tooltip="Pitch angular velocity",
        )
        self.z_label = TooltipLabel(
            text=self.texts["z"].format("--"),
            tooltip="Yaw angular velocity",
        )

        self.layout.addWidget(self.x_label)
        self.layout.addWidget(self.y_label)
        self.layout.addWidget(self.z_label)
        self.layout.addStretch(1)

    def update(self, gyro: Tuple[float, float, float]):
        x_str = "{:6.2f}".format(gyro[0]).replace(" ", "&nbsp;")
        y_str = "{:6.2f}".format(gyro[1]).replace(" ", "&nbsp;")
        z_str = "{:6.2f}".format(gyro[2]).replace(" ", "&nbsp;")

        self.x_label.setText(self.texts["x"].format(x_str))
        self.y_label.setText(self.texts["y"].format(y_str))
        self.z_label.setText(self.texts["z"].format(z_str))


class GyroPlotWidget(PlotWidget):
    def __init__(self):
        super().__init__()
        self.setBackground(Colors.FOREGROUND)
        self.getPlotItem().setTitle("Gyroscope Angular Velocity [°/s]")
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.3)

        self._x_data = np.zeros(DATA_QUEUE_SIZE)
        self._y_data = np.zeros(DATA_QUEUE_SIZE)
        self._z_data = np.zeros(DATA_QUEUE_SIZE)
        self._n_steps = np.array(PLOT_TIME_STEPS)

        self._x_color = Colors.RED
        self._y_color = Colors.GREEN
        self._z_color = Colors.PASTEL_BLUE

        self._update_counter = 0
        self._update_frequency = 1

        self._setup_axes()
        self._setup_legend()
        self._setup_plots()

    def _setup_axes(self):
        self.setYRange(-2.0, 2.0, padding=0)

        # def format_ticks_align_left(values, scale, spacing):
        #     return [f"{v:<4.0f}" for v in values]

        # def format_ticks_align_right(values, scale, spacing):
        #     return [f"{v:4.0f}" for v in values]

        self.text_pen = mkPen(Colors.ON_ACCENT)
        self.getAxis("left").setPen(self.text_pen)
        self.getAxis("left").setTextPen(self.text_pen)
        # self.getAxis("left").tickStrings = format_ticks_align_right

        self.getAxis("bottom").setTicks(STEP_TICKS)
        self.getAxis("bottom").setPen(self.text_pen)
        self.getAxis("bottom").setTextPen(self.text_pen)

        # self.showAxis("right")
        # self.getAxis("right").setPen(self.text_pen)
        # self.getAxis("right").setTextPen(self.text_pen)
        # self.getAxis("right").tickStrings = format_ticks_align_left

    def _setup_legend(self):
        self.legend = self.getPlotItem().addLegend()
        self.legend.anchor(itemPos=(0, 0), parentPos=(0, 1), offset=(15, -35))
        self.legend.setBrush(QBrush(QColor(Colors.ACCENT)))
        self.legend.setPen(mkPen(color=Colors.ON_FOREGROUND, width=0.5))
        self.legend.layout.setContentsMargins(3, 1, 3, 1)
        self.legend.setColumnCount(3)

    def _setup_plots(self):
        x_pen = mkPen(self._x_color, style=Qt.PenStyle.SolidLine, width=1)
        self._x_plot = self.plot(name="X", pen=x_pen, antialias=True, skipFiniteCheck=True)
        self._x_plot.setData(self._n_steps, self._z_data)

        y_pen = mkPen(self._y_color, style=Qt.PenStyle.SolidLine, width=1)
        self._y_plot = self.plot(name="Y", pen=y_pen, antialias=True, skipFiniteCheck=True)
        self._y_plot.setData(self._n_steps, self._x_data)

        z_pen = mkPen(self._z_color, style=Qt.PenStyle.SolidLine, width=1)
        self._z_plot = self.plot(name="Z", pen=z_pen, antialias=True, skipFiniteCheck=True)
        self._z_plot.setData(self._n_steps, self._y_data)


    def update(self, gyro: Tuple[float, float, float]):
        self._x_data = np.roll(self._x_data, -1)
        self._x_data[-1] = gyro[0]

        self._z_data = np.roll(self._z_data, -1)
        self._z_data[-1] = gyro[1]

        self._y_data = np.roll(self._y_data, -1)
        self._y_data[-1] = gyro[2]

        if self._update_counter == 0:
            self._y_plot.setData(self._n_steps, self._x_data)
            self._x_plot.setData(self._n_steps, self._z_data)
            self._z_plot.setData(self._n_steps, self._y_data)

        self._update_counter = (self._update_counter + 1) % self._update_frequency
