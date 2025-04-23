import numpy as np
from typing import Iterable

from PySide6.QtGui import QColor, QBrush
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout

from pyqtgraph import PlotWidget, ScatterPlotItem, PlotCurveItem, TextItem, mkPen, mkBrush

from modules.ui.presets import Colors, TooltipLabel
from modules.ui.plots import (
    DATA_QUEUE_SIZE,
    PLOT_TIME_STEPS,
    STEP_TICKS,
    PlotStatsWidget,
    ENCODER_RAW2DEG,
)


class EncoderWidget(QWidget):
    def __init__(self, name: str):
        super().__init__()
        self.radial_plot_widget = EncoderRadialPlotWidget(name=name)
        self.line_plot_widget = EncoderLinePlotWidget(name=name)
        self.stats_widget = EncoderPlotStatsWidget()

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # plot_layout = QHBoxLayout()
        # plot_layout.addWidget(self.line_plot_widget, stretch=2)
        # plot_layout.addWidget(self.radial_plot_widget, stretch=1)

        layout.addWidget(self.radial_plot_widget, stretch=1)
        # layout.addLayout(plot_layout, stretch=1)
        layout.addWidget(self.stats_widget)

    def update(self, readings: Iterable):
        self.radial_plot_widget.update(readings)
        # self.line_plot_widget.update(readings)


class EncoderPlotStatsWidget(PlotStatsWidget):
    def __init__(self):
        super().__init__()
        html_colored_number = "<span style='color: {}; font-weight: bold; font-family: Courier New, monospace;'>{{}}</span>"
        html_bold = "<span style='font-weight: bold;'>{}</span>"

        self.texts = {
            "angle": f"Msr:{html_colored_number.format(Colors.GREEN)}",
            "angle_change": f"Δ°:{html_colored_number.format(Colors.GREEN)}",
            "magnitude_avg": f"Avg:{html_colored_number.format(Colors.ON_ACCENT)}",
            "magnitude_avg_change": f"Δ:{html_colored_number.format(Colors.ON_ACCENT)}",
        }

        angle_description_label = TooltipLabel(html_bold.format("Angle"))
        self.angle_label = TooltipLabel(self.texts["angle"].format("--"))
        self.angle_change_label = TooltipLabel(self.texts["angle_change"].format("--"))

        magnitude_description_label = TooltipLabel(html_bold.format("Magnitude"))
        self.magnitude_avg_label = TooltipLabel(self.texts["magnitude_avg"].format("--"))
        self.magnitude_avg_change_label = TooltipLabel(
            self.texts["magnitude_avg_change"].format("--")
        )

        self.layout.addWidget(angle_description_label)
        self.layout.addWidget(self.angle_label)
        self.layout.addWidget(self.angle_change_label)
        self.layout.addStretch(1)
        self.layout.addWidget(magnitude_description_label)
        self.layout.addWidget(self.magnitude_avg_label)
        self.layout.addWidget(self.magnitude_avg_change_label)

    def update(
        self, angle_deg: float, angle_deg_change: float, mag_avg: float, mag_avg_change: float
    ):
        ang_str = "{:7.2f}".format(angle_deg).replace(" ", "&nbsp;")
        ang_change_str = "{:7.2f}".format(angle_deg_change).replace(" ", "&nbsp;")
        mag_change_str = "{:6.2f}".format(mag_avg_change).replace(" ", "&nbsp;")
        mag_str = "{:6.2f}".format(mag_avg).replace(" ", "&nbsp;")

        self.angle_label.setText(self.texts["angle"].format(ang_str))
        self.angle_change_label.setText(self.texts["angle_change"].format(ang_change_str))
        self.magnitude_avg_change_label.setText(
            self.texts["magnitude_avg_change"].format(mag_change_str)
        )
        self.magnitude_avg_label.setText(self.texts["magnitude_avg"].format(mag_str))


class EncoderLinePlotWidget(PlotWidget):
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)
        self.setBackground(Colors.FOREGROUND)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setTitle(f"Angle [°] | Magnitude ({self.name})")

        self._angle_data_encoder_freq = np.zeros(DATA_QUEUE_SIZE)
        self._angle_data_fusion_freq = np.zeros(DATA_QUEUE_SIZE)
        self._magnitude_data_encoder_freq = np.zeros(DATA_QUEUE_SIZE)
        self._magnitude_data_fusion_freq = np.zeros(DATA_QUEUE_SIZE)
        self._x = np.array(PLOT_TIME_STEPS)

        self._last_angle_deg = 0
        self._last_magnitude = 0

        self._angle_color = Colors.GREEN

        self._setup_axes()
        self._setup_legend()
        self._setup_plots()

    def update(self, readings: Iterable):
        if not readings:
            return

        # TODO: convert RAW2DEG on client's side
        angle_deg_changes = []
        magnitude_changes = []

        for data in readings:
            angle_deg = data.position * ENCODER_RAW2DEG
            angle_deg_change = (angle_deg - self._last_angle_deg + 180) % 360 - 180
            angle_deg_changes.append(abs(angle_deg_change))
            self._last_angle_deg = angle_deg

            magnitude_change = data.magnitude - self._last_magnitude
            magnitude_changes.append(magnitude_change)
            self._last_magnitude = data.magnitude

            self._angle_data_encoder_freq = np.roll(self._angle_data_encoder_freq, -1)
            self._angle_data_encoder_freq[-1] = angle_deg
            self._magnitude_data_encoder_freq = np.roll(self._magnitude_data_encoder_freq, -1)
            self._magnitude_data_encoder_freq[-1] = data.magnitude

        self._angle_data_fusion_freq = np.roll(self._angle_data_fusion_freq, -1)
        self._angle_data_fusion_freq[-1] = self._last_angle_deg
        self._magnitude_data_fusion_freq = np.roll(self._magnitude_data_fusion_freq, -1)
        self._magnitude_data_fusion_freq[-1] = self._last_magnitude

        self._angle_plot.setData(self._x, np.cos(np.deg2rad(self._angle_data_encoder_freq)))

    def _setup_axes(self):
        self.setYRange(1.5, -1.5, padding=0)

        self.text_pen = mkPen(Colors.ON_ACCENT)
        self.getAxis("left").setPen(self.text_pen)
        self.getAxis("left").setTextPen(self.text_pen)

        self.getAxis("bottom").setTicks(STEP_TICKS)
        self.getAxis("bottom").setPen(self.text_pen)
        self.getAxis("bottom").setTextPen(self.text_pen)

        # self.showAxis("right")
        # self.getAxis("right").setPen(self.text_pen)
        # self.getAxis("right").setTextPen(self.text_pen)

    def _setup_legend(self):
        self.legend = self.getPlotItem().addLegend()
        self.legend.anchor(itemPos=(0, 0), parentPos=(0, 1), offset=(15, -35))
        self.legend.setBrush(QBrush(QColor(Colors.ACCENT)))
        self.legend.setPen(mkPen(color=Colors.ON_FOREGROUND, width=0.5))
        self.legend.layout.setContentsMargins(3, 1, 3, 1)
        # self.legend.setColumnCount(3)

    def _setup_plots(self):
        angle_pen = mkPen(self._angle_color, style=Qt.PenStyle.SolidLine, width=2)
        self._angle_plot = self.plot(name="Measured", pen=angle_pen, antialias=True)
        self._angle_plot.setData(self._x, self._angle_data_fusion_freq)


class EncoderRadialPlotWidget(PlotWidget):
    def __init__(self, name: str, *args, **kwargs):
        self.name = name

        super().__init__(*args, **kwargs)
        self.setBackground(Colors.FOREGROUND)
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setTitle(f"Angle [°] | Magnitude ({self.name})")

        self.max_samples_fusion_freq = 15
        self.xs_fusion_freq = np.zeros(self.max_samples_fusion_freq)
        self.ys_fusion_freq = np.zeros(self.max_samples_fusion_freq)

        self.last_angle_deg = 0
        self.last_magnitude = 0

        self._setup_axes()
        self._setup_reference_circle()
        self._setup_plots()
        self._setup_history_samples_color_brushes()

    def update(self, readings: Iterable):
        if not readings:
            return

        # TODO: convert RAW2DEG on client's side
        # TODO: aggregate on client's side

        angle_deg_changes = []
        magnitude_changes = []
        magnitude_sum = 0

        for data in readings:
            angle_deg = data.position * ENCODER_RAW2DEG
            angle_deg_change = (angle_deg - self.last_angle_deg + 180) % 360 - 180
            angle_deg_changes.append(abs(angle_deg_change))
            self.last_angle_deg = angle_deg

            magnitude_change = data.magnitude - self.last_magnitude
            magnitude_changes.append(magnitude_change)
            magnitude_sum += data.magnitude
            self.last_magnitude = data.magnitude

        angle = np.deg2rad(self.last_angle_deg)
        self.xs_fusion_freq = np.roll(self.xs_fusion_freq, -1)
        self.ys_fusion_freq = np.roll(self.ys_fusion_freq, -1)
        self.xs_fusion_freq[-1] = self.last_magnitude * np.cos(angle)
        self.ys_fusion_freq[-1] = self.last_magnitude * np.sin(angle)

        angle_deg_change = sum(angle_deg_changes)
        avg_magnitude = magnitude_sum / len(readings)
        avg_magnitude_change = sum(magnitude_changes) / len(magnitude_changes)

        # Update plots
        self.current_point.setData([self.xs_fusion_freq[-1]], [self.ys_fusion_freq[-1]])
        self.points_fusion_freq.setData(
            x=self.xs_fusion_freq, y=self.ys_fusion_freq, brush=self.color_brushes_fusion_freq
        )

        # Update stats
        self.parent().stats_widget.update(
            self.last_angle_deg, angle_deg_change, avg_magnitude, avg_magnitude_change
        )

    def _setup_axes(self):
        self.text_pen = mkPen(Colors.ON_ACCENT)

        self.getAxis("left").setPen(self.text_pen)
        self.getAxis("left").setTextPen(self.text_pen)
        # self.hideAxis("left")

        self.getAxis("bottom").setPen(self.text_pen)
        self.getAxis("bottom").setTextPen(self.text_pen)

        # self.getAxis("right").setPen(self.text_pen)
        # self.getAxis("right").setTextPen(self.text_pen)
        # self.showAxis("right")

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

            # Grid line angle text
            text = TextItem(str(angle), anchor=(0.5, 0.5), color=Colors.ON_ACCENT_DIM)
            text_offset = radius + 10
            text.setPos(text_offset * np.cos(rad), text_offset * np.sin(rad))
            self.addItem(text)

    def _setup_plots(self):
        self.current_point = ScatterPlotItem(size=10, pen=mkPen(Colors.GREEN, width=3))
        self.addItem(self.current_point)
        self.points_fusion_freq = ScatterPlotItem(size=5, pen=None)
        self.addItem(self.points_fusion_freq)

    def _setup_history_samples_color_brushes(self):
        base_color_fusion_freq = QColor(Colors.GREEN)
        base_color_fusion_freq.setAlpha(200)
        self.color_brushes_fusion_freq = []

        for i in range(self.max_samples_fusion_freq):
            i_color_ff = QColor(base_color_fusion_freq)
            i_color_ff.setAlpha(
                int(base_color_fusion_freq.alpha() * i / self.max_samples_fusion_freq)
            )
            self.color_brushes_fusion_freq.append(mkBrush(i_color_ff))
