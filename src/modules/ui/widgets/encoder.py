
import numpy as np
from typing import Iterable

from PySide6.QtGui import QColor
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout

from pyqtgraph import PlotWidget, ScatterPlotItem, PlotCurveItem, TextItem, mkPen, mkBrush

from modules.ui.presets import Colors, CustomTooltipLabel, MColors
from modules.ui.plots import PlotStatsWidget, ENCODER_RAW2DEG


class EncoderWidget(QWidget):
    def __init__(self, name: str):
        super().__init__()
        self.plot_widget = EncoderPlotWidget(name=name)
        self.stats_widget = EncoderPlotStatsWidget()

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget, stretch=1)
        layout.addWidget(self.stats_widget)

    def update(self, readings: Iterable):
        self.plot_widget.update(readings)


class EncoderPlotStatsWidget(PlotStatsWidget):
    def __init__(self):
        super().__init__()
        html_colored_number = "<span style='color: {}; font-weight: bold; font-family: Courier New, monospace;'>{{}}</span>"

        self.texts = {
            "angle": f"Msr:{html_colored_number.format(Colors.GREEN)}",
            "angle_change": f"Δ°:{html_colored_number.format(Colors.GREEN)}",
            "mag_avg": f"Avg:{html_colored_number.format(Colors.ON_ACCENT)}",
            "mag_std": f"Std:{html_colored_number.format(Colors.ON_ACCENT)}"
        }

        angle_description_label = CustomTooltipLabel(text="<span style='font-weight: bold;'>Angle</span>")
        self.angle_label = CustomTooltipLabel(text=self.texts["angle"].format("--"))
        self.angle_change_label = CustomTooltipLabel(text=self.texts["angle_change"].format("--"))
        
        mag_description_label = CustomTooltipLabel(text="<span style='font-weight: bold;'>Magnitude</span>")        
        self.mag_avg_label = CustomTooltipLabel(text=self.texts["mag_avg"].format("--"))
        self.mag_std_label = CustomTooltipLabel(text=self.texts["mag_std"].format("--"))

        self.layout.addWidget(angle_description_label)
        self.layout.addWidget(self.angle_label)
        self.layout.addWidget(self.angle_change_label)
        self.layout.addStretch(1)
        self.layout.addWidget(mag_description_label)
        self.layout.addWidget(self.mag_avg_label)
        self.layout.addWidget(self.mag_std_label)

    def update(self, angle_deg, angle_change, mag_avg, mag_std):
        angle_str = "{:7.2f}".format(angle_deg).replace(" ", "&nbsp;")
        angle_change_str = "{:7.2f}".format(angle_change).replace(" ", "&nbsp;")
        mag_avg_str = "{:6.2f}".format(mag_avg).replace(" ", "&nbsp;")
        mag_std_str = "{:6.2f}".format(mag_std).replace(" ", "&nbsp;")

        self.angle_label.setText(self.texts["angle"].format(angle_str))
        self.angle_change_label.setText(self.texts["angle_change"].format(angle_change_str))
        self.mag_avg_label.setText(self.texts["mag_avg"].format(mag_avg_str))
        self.mag_std_label.setText(self.texts["mag_std"].format(mag_std_str))


class EncoderPlotWidget(PlotWidget):
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setBackground(Colors.FOREGROUND)
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setTitle(f"Angle [°] | Magnitude ({name})")

        self.text_pen = mkPen(Colors.ON_ACCENT)
        self._setup_axes()
        self._setup_reference_circle()
        self._setup_plots()

        # Data storage
        self.max_samples = 50
        self.angles = np.zeros(self.max_samples)
        self.magnitudes = np.zeros(self.max_samples)
        self.xs = np.zeros(self.max_samples)
        self.ys = np.zeros(self.max_samples)
        self.sample_indices = np.arange(self.max_samples)
        self.next_idx = 0
        self.data_count = 0

        # Computed brushes for previous points
        base_color = QColor(Colors.GREEN)
        base_color.setAlpha(128)
        self.color_brushes = []
        for i in range(self.max_samples):
            i_color = QColor(base_color)
            i_color.setAlpha(int(base_color.alpha() * i / self.max_samples))
            self.color_brushes.append(mkBrush(i_color))

        # Precomputed radial plot positions
        theta = np.linspace(0, 2 * np.pi, 100)
        self.circle_x = np.cos(theta)
        self.circle_y = np.sin(theta)

        self.rolling_sum = 0
        self.rolling_sum_sq = 0

        self.update_counter = 0

        self.deg = 0
        self.angle_std = 0
        self.avg_angle_change = 0
        self.mag_avg = 0
        self.mag_std = 0

    def _setup_axes(self):
        self.getAxis("left").setPen(self.text_pen)
        self.getAxis("left").setTextPen(self.text_pen)
        self.getAxis("bottom").setPen(self.text_pen)
        self.getAxis("bottom").setTextPen(self.text_pen)

    def _setup_reference_circle(self):
        # Radial grid circle
        radius = 35
        theta = np.linspace(0, 2 * np.pi, 100)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        self.circle = ScatterPlotItem(size=1, pen=mkPen(Colors.ON_ACCENT_DIM), brush=None)
        self.circle.setData(x=x, y=y)
        self.addItem(self.circle)

        # Radial grid lines
        for angle in range(0, 360, 30):
            rad = np.deg2rad(angle)
            x = [0, radius * np.cos(rad)]
            y = [0, radius * np.sin(rad)]
            line = PlotCurveItem(
                x=x, y=y, pen=mkPen(Colors.ON_ACCENT_DIM, width=0.5, style=Qt.DotLine)
            )
            self.addItem(line)

            # Grid line angle
            text = TextItem(str(angle), anchor=(0.5, 0.5), color=Colors.ON_ACCENT_DIM)
            text_offset = radius + 10
            text.setPos(text_offset * np.cos(rad), text_offset * np.sin(rad))
            self.addItem(text)

    def _setup_plots(self):
        self.current_point = ScatterPlotItem(size=10, pen=mkPen(Colors.GREEN, width=3))
        self.addItem(self.current_point)
        self.points = ScatterPlotItem(size=5, pen=None)
        self.addItem(self.points)

    def update(self, readings: Iterable):
        if not readings:
            return
            
        last_data = readings[0]
        angle_changes = []
        
        for data in readings:
            # Position to deg and coordinates 
            self.deg = data.position * ENCODER_RAW2DEG
            rad = np.deg2rad(self.deg)
            x, y = data.magnitude * np.cos(rad), data.magnitude * np.sin(rad)
            
            # Angle change with wrap-around
            prev_angle = last_data.position * ENCODER_RAW2DEG
            change = (self.deg - prev_angle + 180) % 360 - 180
            angle_changes.append(abs(change))
            last_data = data
            
            # Magnitude rollin sum
            if self.data_count >= self.max_samples:
                old_mag = self.magnitudes[self.next_idx]
                self.rolling_sum -= old_mag
                self.rolling_sum_sq -= old_mag**2
                
            self.rolling_sum += data.magnitude
            self.rolling_sum_sq += data.magnitude**2
            
            # Store data in circular buffer
            self.angles[self.next_idx] = self.deg
            self.magnitudes[self.next_idx] = data.magnitude
            self.xs[self.next_idx] = x
            self.ys[self.next_idx] = y
            
            # Update indices
            self.next_idx = (self.next_idx + 1) % self.max_samples
            self.data_count = min(self.data_count + 1, self.max_samples)
        
        self.avg_angle_change = sum(angle_changes) / len(angle_changes)
        self.current_point.setData([x], [y])
        
        if self.data_count < self.max_samples:
            x_plot = self.xs[:self.data_count]
            y_plot = self.ys[:self.data_count]
        else:
            x_plot = np.roll(self.xs, -self.next_idx)
            y_plot = np.roll(self.ys, -self.next_idx)  
        
        self.points.setData(x=x_plot, y=y_plot, brush=self.color_brushes[:len(x_plot)])
        
        self.mag_avg = self.rolling_sum / self.data_count
        self.mag_std = np.sqrt(max(0, (self.rolling_sum_sq / self.data_count) - (self.mag_avg**2)))
            
        parent = self.parent()
        if parent and hasattr(parent, "stats_widget"):
            parent.stats_widget.update(self.deg, self.avg_angle_change, self.mag_avg, self.mag_std)