import numpy as np

from PySide6.QtGui import QColor, QBrush
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout 

from pyqtgraph import PlotWidget, ScatterPlotItem, PlotCurveItem, TextItem, mkPen, mkBrush

from modules.ui.data import EncoderPlotData
from modules.ui.presets import UIColors, TooltipLabel
from modules.ui.plots import (
    STEP_TICKS,
    PlotStatsWidget,
)


class EncoderWidget(QWidget):
    def __init__(self, name: str):
        super().__init__()
        self.radial_plot_widget = EncoderRadialPlotWidget(name=name)
        self.stats_widget = EncoderPlotStatsWidget()

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.radial_plot_widget, stretch=1)
        layout.addWidget(self.stats_widget)

    def update(self, data: EncoderPlotData):
        self.radial_plot_widget.update(data.radial_xs, data.radial_ys)
        self.stats_widget.update(data.angle_deg, data.angle_deg_change, data.avg_magnitude, data.avg_magnitude_change)

class EncoderPlotStatsWidget(PlotStatsWidget):
    def __init__(self):
        super().__init__(QVBoxLayout)
        html_colored_number = "<span style='color: {}; font-weight: bold; font-family: Courier New, monospace;'>{{}}</span>"
        html_bold = "<span style='font-weight: bold;'>{}</span>"

        self.texts = {
            "angle": f"Msr:{html_colored_number.format(UIColors.GREEN)}",
            "angle_change": f"Δ°:{html_colored_number.format(UIColors.GREEN)}",
            "magnitude_avg": f"Avg:{html_colored_number.format(UIColors.ON_ACCENT)}",
            "magnitude_avg_change": f"Δ :{html_colored_number.format(UIColors.ON_ACCENT)}",
        }

        angle_description_label = TooltipLabel(html_bold.format("Angle"))
        self.angle_label = TooltipLabel(self.texts["angle"].format("--"))
        self.angle_change_label = TooltipLabel(self.texts["angle_change"].format("--"))

        magnitude_description_label = TooltipLabel(html_bold.format("Mag"))
        self.magnitude_avg_label = TooltipLabel(self.texts["magnitude_avg"].format("--"))
        self.magnitude_avg_change_label = TooltipLabel(
            self.texts["magnitude_avg_change"].format("--")
        )

        angle_layout = QHBoxLayout()
        angle_layout.addWidget(angle_description_label)
        angle_layout.addStretch(1)
        angle_layout.addWidget(self.angle_label)
        angle_layout.addWidget(self.angle_change_label)
        mag_layout = QHBoxLayout()        
        mag_layout.addWidget(magnitude_description_label)
        mag_layout.addStretch(1)
        mag_layout.addWidget(self.magnitude_avg_label)
        mag_layout.addWidget(self.magnitude_avg_change_label)
        self.layout.addLayout(angle_layout)
        self.layout.addLayout(mag_layout)

    def update(
        self, angle_deg: float, angle_deg_change: float, mag_avg: float, mag_avg_change: float
    ):
        ang_str = "{:7.2f}".format(angle_deg).replace(" ", "&nbsp;")
        ang_change_str = "{:7.2f}".format(angle_deg_change).replace(" ", "&nbsp;")
        mag_change_str = "{:7.2f}".format(mag_avg_change).replace(" ", "&nbsp;")
        mag_str = "{:7.2f}".format(mag_avg).replace(" ", "&nbsp;")

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
        self.setBackground(UIColors.FOREGROUND)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setTitle(f"Angle [°] | Magnitude ({self.name})")
        self.setAspectLocked(True)
        self.setMinimumSize(300, 200)
        

        self._angle_color = UIColors.GREEN

        self._setup_axes()
        self._setup_legend()
        self._setup_plots()

    def update(self, radial_xs: np.ndarray):
        self._angle_plot.setData(self._x, radial_xs)

    def _setup_axes(self):
        self.setYRange(1.5, -1.5, padding=0)

        self.text_pen = mkPen(UIColors.ON_ACCENT)
        self.getAxis("left").setPen(self.text_pen)
        self.getAxis("left").setTextPen(self.text_pen)

        self.getAxis("bottom").setTicks(STEP_TICKS)
        self.getAxis("bottom").setPen(self.text_pen)
        self.getAxis("bottom").setTextPen(self.text_pen)

    def _setup_legend(self):
        self.legend = self.getPlotItem().addLegend()
        self.legend.anchor(itemPos=(0, 0), parentPos=(0, 1), offset=(15, -35))
        self.legend.setBrush(QBrush(QColor(UIColors.ACCENT)))
        self.legend.setPen(mkPen(color=UIColors.ON_FOREGROUND, width=0.5))
        self.legend.layout.setContentsMargins(3, 1, 3, 1)
        self.legend.setColumnCount(3)

    def _setup_plots(self):
        angle_pen = mkPen(self._angle_color, style=Qt.PenStyle.SolidLine, width=2)
        self._angle_plot = self.plot(name="Measured", pen=angle_pen, antialias=True)
        self._angle_plot.setData(self._x, self._angle_data_fusion_freq)


class EncoderRadialPlotWidget(PlotWidget):
    def __init__(self, name: str, *args, **kwargs):
        self.name = name

        super().__init__(*args, **kwargs)
        self.setBackground(UIColors.FOREGROUND)
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setTitle(f"Angle [°] | Magnitude ({self.name})")

        self._setup_axes()
        self._setup_reference_circle()
        self._setup_plots()
        self._setup_history_samples_color_brushes()

    def update(self, radial_xs: np.ndarray, radial_ys: np.ndarray):
        self._points.setData(x=radial_xs, y=radial_ys, brush=self._points_brushes)

    def _setup_axes(self):
        self.setYRange(-50, 50, padding=0)
        self.setXRange(-50, 50, padding=0)

        self.text_pen = mkPen(UIColors.ON_ACCENT)

        self.getAxis("left").setPen(self.text_pen)
        self.getAxis("left").setTextPen(self.text_pen)

        self.getAxis("bottom").setPen(self.text_pen)
        self.getAxis("bottom").setTextPen(self.text_pen)

    def _setup_reference_circle(self):
        # Radial grid circle
        radius = 35
        theta = np.linspace(0, 2 * np.pi, 30)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        self.circle = PlotCurveItem(pen=mkPen(UIColors.ON_ACCENT_DIM))
        self.circle.setData(x=x, y=y)
        self.addItem(self.circle)

        # Radial grid lines
        for angle in range(0, 360, 30):
            rad = np.deg2rad(angle)
            x = [0, radius * np.cos(rad)]
            y = [0, radius * np.sin(rad)]
            line = PlotCurveItem(x=x, y=y, pen=mkPen(UIColors.ON_ACCENT_DIM))
            self.addItem(line)

            # Grid line angle text
            text = TextItem(str(angle), anchor=(0.5, 0.5), color=UIColors.ON_ACCENT_DIM)
            text_offset = radius + 10
            text.setPos(text_offset * np.cos(rad), text_offset * np.sin(rad))
            self.addItem(text)

    def _setup_plots(self):
        self.current_point = ScatterPlotItem(size=10, pen=mkPen(UIColors.GREEN, width=3))
        self.addItem(self.current_point)
        self._points = ScatterPlotItem(size=5, pen=None)
        self.addItem(self._points)

    def _setup_history_samples_color_brushes(self):
        base_color = QColor(UIColors.ON_PRIMARY)
        base_color.setAlpha(255)
        self._points_brushes = []

        for i in range(EncoderPlotData.HISTORY_SIZE):
            i_color_ff = QColor(base_color)
            i_color_ff.setAlpha(
                int(base_color.alpha() * i / EncoderPlotData.HISTORY_SIZE)
            )
            self._points_brushes.append(mkBrush(i_color_ff))
