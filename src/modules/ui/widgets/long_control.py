import numpy as np
from collections import deque

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout

from pyqtgraph import PlotWidget, ScatterPlotItem, PlotCurveItem, mkPen
import pyqtgraph as pg

from modules.ui.plots import PlotStatsWidget, DATA_QUEUE_SIZE, STEP_TICKS, PLOT_TIME_STEPS
from modules.ui.presets import Colors

from time import time

class LongitudinalControlWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.plot_widget = SpeedPlotWidget()
        self.stats_widget = SpeedPlotStatsWidget()

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget, stretch=1)
        layout.addWidget(self.stats_widget)

    def update(self, measured_speed: float, target_speed: float, engine_power_percent: float):
        self.plot_widget.update(measured_speed, target_speed, engine_power_percent)
        self.stats_widget.update(measured_speed, target_speed, engine_power_percent)


class SpeedPlotStatsWidget(PlotStatsWidget):
    def __init__(self):
        super().__init__()

        html_colored_number = "<span style='color: {}; font-weight: bold; font-family: Courier New, monospace;'>{{}}</span>"

        self.texts = {
            "speed": f"Measured: {html_colored_number.format(Colors.GREEN)} [m/s]",
            "target": f"Target: {html_colored_number.format(Colors.GREEN)} [m/s]",
            "error": f"Error: {html_colored_number.format(Colors.ON_ACCENT)} [m/s]",
            "power": f"Power: {html_colored_number.format(Colors.ORANGE)} [%]",
        }

        # Left
        self.speed_label = QLabel(self.texts["speed"].format(0.0))
        self.target_label = QLabel(self.texts["target"].format(0.0))
        self.error_label = QLabel(self.texts["error"].format(0.0))

        # Right
        self.power_label = QLabel(self.texts["power"].format(0.0))
        self.power_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.layout.addWidget(self.speed_label)
        self.layout.addWidget(self.target_label)
        self.layout.addWidget(self.error_label)
        self.layout.addStretch(1)
        self.layout.addWidget(self.power_label)

    def update(self, measured_speed: float, target_speed: float, engine_power_percent: float):
        ms_str = "{:+6.2f}".format(measured_speed).replace(" ", "&nbsp;")
        ts_str = "{:+6.2f}".format(target_speed).replace(" ", "&nbsp;")
        err_str = "{:+6.2f}".format(measured_speed - target_speed).replace(" ", "&nbsp;")
        ep_str = "{:+6.2f}".format(engine_power_percent).replace(" ", "&nbsp;")

        self.speed_label.setText(self.texts["speed"].format(ms_str))
        self.target_label.setText(self.texts["target"].format(ts_str))
        self.error_label.setText(self.texts["error"].format(err_str))
        self.power_label.setText(self.texts["power"].format(ep_str))


class SpeedPlotWidget(PlotWidget):
    def paintEvent(self, ev):
        t = time()
        super().paintEvent(ev)
        print(f"\r{time() - t}", end="")
        
    def __init__(self):
        super().__init__()
        
        self.setBackground(Colors.FOREGROUND)
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
        self.getPlotItem().setTitle("Speed | Power")
        
        self.speed_color = Colors.GREEN
        self.power_color = Colors.ORANGE
        self.text_pen = mkPen(Colors.ON_ACCENT)
        
        self._speed_data = np.zeros(DATA_QUEUE_SIZE)
        self._target_data = np.zeros(DATA_QUEUE_SIZE)
        self._power_data = np.zeros(DATA_QUEUE_SIZE)
        self._x_data = np.array(PLOT_TIME_STEPS)
        
        self._update_counter= 0
        self._update_frequency = 2

        # Y-axis rescaling
        self._rescale_counter = 0
        self._rescale_frequency = 10
        self._rescale_min_range = 2.0
        self._rescale_y_padding = 1.0

        self._setup_axes()
        self._setup_legend()
        self._setup_plots()


    def _setup_axes(self):
        self.setAutoVisible(y=False)
        self.setYRange(-self._rescale_min_range/2, self._rescale_min_range/2, padding=self._rescale_y_padding)
 
        self.getAxis("left").setPen(self.text_pen)
        self.getAxis("left").setTextPen(self.text_pen)
        
        self.getAxis("bottom").setTicks(STEP_TICKS)
        self.getAxis("bottom").setPen(self.text_pen)
        self.getAxis("bottom").setTextPen(self.text_pen)
        
        self.power_y_axis = pg.AxisItem("right")
        self.power_y_axis.setPen(self.text_pen)
        self.power_y_axis.setTextPen(self.text_pen)
        self.getPlotItem().layout.addItem(self.power_y_axis, 2, 3)
        
        self._power_viewbox = pg.ViewBox()
        self.power_y_axis.linkToView(self._power_viewbox)
        self._power_viewbox.setXLink(self.getPlotItem())
        self._power_viewbox.setYRange(-100, 100, padding=0)
        self.getPlotItem().scene().addItem(self._power_viewbox)
        
        # Connect power and speed viewboxes
        self.getPlotItem().vb.sigResized.connect(self._update_views)
        

    def _setup_legend(self):
        self.legend = self.getPlotItem().addLegend()
        self.legend.anchor(itemPos=(0, 0), parentPos=(0, 1), offset=(15, -35))
        self.legend.setBrush(QBrush(QColor(Colors.ACCENT)))
        self.legend.setPen(mkPen(color=Colors.ON_FOREGROUND, width=0.5))
        self.legend.layout.setContentsMargins(3, 1, 3, 1)
        self.legend.setColumnCount(3)

    def _setup_plots(self):
        speed_pen = mkPen(self.speed_color, style=Qt.PenStyle.SolidLine, width=2)
        self._measured_speed_plot = self.plot(name="Measured", pen=speed_pen, antialias=True)
        self._measured_speed_plot.setData(self._x_data, self._speed_data)
        
        target_pen = mkPen(self.speed_color, style=Qt.PenStyle.DashLine, width=2)
        self._target_speed_plot = self.plot(name="Target", pen=target_pen, antialias=True)
        self._target_speed_plot.setData(self._x_data, self._target_data)
        
        power_pen = mkPen(self.power_color, style=Qt.PenStyle.SolidLine, width=1)
        self._power_plot = pg.PlotCurveItem(name="Power", pen=power_pen, antialias=True)
        self._power_plot.setData(self._x_data, self._power_data)
        self._power_viewbox.addItem(self._power_plot)
        self.legend.addItem(PlotCurveItem(pen=power_pen), "Power")
        
    def _update_views(self):
        """Keeps power and main viewboxes synced on resize"""
        self._power_viewbox.setGeometry(self.getPlotItem().vb.sceneBoundingRect())

    def update(self, measured_speed: float, target_speed: float, engine_power_percent: float):
        self._speed_data = np.roll(self._speed_data, -1)
        self._target_data = np.roll(self._target_data, -1)
        self._power_data = np.roll(self._power_data, -1)
        
        self._speed_data[-1] = measured_speed
        self._target_data[-1] = target_speed
        self._power_data[-1] = engine_power_percent
        
        if self._update_counter == 0:
            self._measured_speed_plot.setData(self._x_data, self._speed_data)
            self._target_speed_plot.setData(self._x_data, self._target_data)
            self._power_plot.setData(self._x_data, self._power_data)
            
            if self._rescale_counter == 0:
                self._update_y_scale()
                self._rescale_counter = (self._rescale_counter + 1) % self._rescale_frequency
            
        self._update_counter = (self._update_counter + 1) % self._update_frequency

    
    def _update_y_scale(self):
        data_min = np.min(self._target_data)
        data_max = np.max(self._target_data)
        
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