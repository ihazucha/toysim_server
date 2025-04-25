import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QWidget, QVBoxLayout

from pyqtgraph import PlotWidget, mkPen

from modules.ui.plots import PlotStatsWidget, DATA_QUEUE_SIZE, STEP_TICKS, PLOT_TIME_STEPS
from modules.ui.presets import Colors, TooltipLabel, PlotWidgetHorizontalCursor


class LateralControlWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.plot_widget = SteeringPlotWidget()
        self.stats_widget = SteeringPlotStatsWidget()

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget, stretch=1)
        layout.addWidget(self.stats_widget)

    def update(self, estimated: float, target: float, input: float):
        self.plot_widget.update(estimated, target, input)
        self.stats_widget.update(estimated, target, input)


class SteeringPlotStatsWidget(PlotStatsWidget):
    def __init__(self):
        super().__init__()

        html_colored_number = "<span style='color: {}; font-weight: bold; font-family: Courier New, monospace;'>{{}}</span>"

        self.texts = {
            "estimated_wsa": f"Est: {html_colored_number.format(Colors.PASTEL_BLUE)}",
            "target_wsa": f"Tgt: {html_colored_number.format(Colors.PASTEL_BLUE)}",
            "error_wsa": f"Err: {html_colored_number.format(Colors.RED)}",
            "input": f"<span style='font-weight: bold'>SIA</span>: {html_colored_number.format(Colors.ORANGE)}",
        }

        wsa_header_label = TooltipLabel(
            text="<span style='font-weight: bold'>WSA</span>",
            tooltip="Wheel Steer Angle"
        )

        self.estimated_sa_label = TooltipLabel(
            text=self.texts["estimated_wsa"].format("--"),
            tooltip="Estimated (Measured) WSA",
        )
        self.target_sa_label = TooltipLabel(
            text=self.texts["target_wsa"].format("--"),
            tooltip="Target WSA as set by the controller",
        )
        self.error_sa_label = TooltipLabel(
            text=self.texts["error_wsa"].format("--"),
            tooltip="Target - Estimated WSA difference",
        )

        self.input_sa_label = TooltipLabel(
            text=self.texts["input"].format("--"),
            tooltip="Steering Input Angle (e.g. steering wheel or servo angle)",
        )

        self.layout.addWidget(wsa_header_label)
        self.layout.addWidget(self.estimated_sa_label)
        self.layout.addWidget(self.target_sa_label)
        self.layout.addWidget(self.error_sa_label)
        self.layout.addStretch(1)
        self.layout.addWidget(self.input_sa_label)

    def update(self, estimated: float, target: float, input: float):
        est_str = "{:6.2f}".format(estimated).replace(" ", "&nbsp;")
        tgt_str = "{:6.2f}".format(target).replace(" ", "&nbsp;")
        err_str = "{:6.2f}".format(target - estimated).replace(" ", "&nbsp;")
        inp_str = "{:6.2f}".format(input).replace(" ", "&nbsp;")

        self.estimated_sa_label.setText(self.texts["estimated_wsa"].format(est_str))
        self.target_sa_label.setText(self.texts["target_wsa"].format(tgt_str))
        self.error_sa_label.setText(self.texts["error_wsa"].format(err_str))
        self.input_sa_label.setText(self.texts["input"].format(inp_str))


class SteeringPlotWidget(PlotWidget):
    def __init__(self):
        super().__init__()
        self.setBackground(Colors.FOREGROUND)
        self.getPlotItem().setTitle("Wheel Steer Angle | Steering Input Angle [°]")
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.3)

        self._target_sa_data = np.zeros(DATA_QUEUE_SIZE)
        self._input_sa_data = np.zeros(DATA_QUEUE_SIZE)
        self._estimated_sa_data = np.zeros(DATA_QUEUE_SIZE)
        self._x = np.array(PLOT_TIME_STEPS)

        self._estimated_sa_color = Colors.PASTEL_BLUE
        self._target_color = Colors.PASTEL_BLUE
        self._input_color = Colors.ORANGE

        self._update_counter = 0
        self._update_frequency = 1

        self._setup_axes()
        self._setup_legend()
        self._setup_plots()

        self.cursor = PlotWidgetHorizontalCursor(
            self, x_values=self._x, update_values_at_index_callback=self._update_values_at_index
        )

    def _setup_axes(self):
        self.setYRange(-35.0, 35.0, padding=0)

        def format_ticks_align_left(values, scale, spacing):
            return [f"{v:<4.0f}" for v in values]

        def format_ticks_align_right(values, scale, spacing):
            return [f"{v:4.0f}" for v in values]

        self.text_pen = mkPen(Colors.ON_ACCENT)
        self.getAxis("left").setPen(self.text_pen)
        self.getAxis("left").setTextPen(self.text_pen)
        self.getAxis("left").tickStrings = format_ticks_align_right

        self.getAxis("bottom").setTicks(STEP_TICKS)
        self.getAxis("bottom").setPen(self.text_pen)
        self.getAxis("bottom").setTextPen(self.text_pen)

        self.showAxis("right")
        self.getAxis("right").setPen(self.text_pen)
        self.getAxis("right").setTextPen(self.text_pen)
        self.getAxis("right").tickStrings = format_ticks_align_left

    def _setup_legend(self):
        self.legend = self.getPlotItem().addLegend()
        self.legend.anchor(itemPos=(0, 0), parentPos=(0, 1), offset=(15, -35))
        self.legend.setBrush(QBrush(QColor(Colors.ACCENT)))
        self.legend.setPen(mkPen(color=Colors.ON_FOREGROUND, width=0.5))
        self.legend.layout.setContentsMargins(3, 1, 3, 1)
        self.legend.setColumnCount(3)

    def _setup_plots(self):
        estimated_sa_pen = mkPen(self._estimated_sa_color, style=Qt.PenStyle.SolidLine, width=1)
        self._estimated_sa_plot = self.plot(name="Estimated", pen=estimated_sa_pen, antialias=True)
        self._estimated_sa_plot.setData(self._x, self._estimated_sa_data)

        target_sa_pen = mkPen(self._target_color, style=Qt.PenStyle.DashLine, width=1)
        self._target_sa_plot = self.plot(name="Target", pen=target_sa_pen, antialias=True)
        self._target_sa_plot.setData(self._x, self._target_sa_data)

        input_sa_pen = mkPen(self._input_color, style=Qt.PenStyle.SolidLine, width=1)
        self._input_sa_plot = self.plot(name="Input", pen=input_sa_pen, antialias=True)
        self._input_sa_plot.setData(self._x, self._input_sa_data)

    def _update_values_at_index(self, idx):
        parent = self.parent()

        if not (parent and hasattr(parent, "stats_widget")):
            print(
                f"[{self.__class__.__name__}] Plot has no parent or stats widget to show values in"
            )
            return

        estimated = self._estimated_sa_data[idx]
        target = self._target_sa_data[idx]
        input_val = self._input_sa_data[idx]
        parent.stats_widget.update(estimated=estimated, target=target, input=input_val)

    def update(self, estimated: float, target: float, input: float | None = None):
        self._target_sa_data = np.roll(self._target_sa_data, -1)
        self._target_sa_data[-1] = target

        self._estimated_sa_data = np.roll(self._estimated_sa_data, -1)
        self._estimated_sa_data[-1] = estimated

        if input is not None:
            self._input_sa_data = np.roll(self._input_sa_data, -1)
            self._input_sa_data[-1] = input

        if self._update_counter == 0:
            self._target_sa_plot.setData(self._x, self._target_sa_data)
            self._estimated_sa_plot.setData(self._x, self._estimated_sa_data)
            self._input_sa_plot.setData(self._x, self._input_sa_data)

        self._update_counter = (self._update_counter + 1) % self._update_frequency

