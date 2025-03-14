import numpy as np

from collections import deque
from time import time_ns

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QLinearGradient, QBrush, QVector3D, QFont
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame, QVBoxLayout

from pyqtgraph import PlotWidget, ScatterPlotItem, PlotCurveItem, mkPen, mkBrush
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLGridItem

from datalink.data import EncoderData, Rotation
from modules.ui.presets import DefaultMonospaceFont

FPS = 60
DATA_QUEUE_LENGTH_SECONDS = 5
DATA_QUEUE_SIZE = FPS * DATA_QUEUE_LENGTH_SECONDS

PLOT_QUEUE_DEFAULT_DATA = list([0 for _ in range(DATA_QUEUE_SIZE)])
PLOT_TIME_STEPS = np.arange(-DATA_QUEUE_SIZE, 0, 1)
STEP_MAJOR_TICKS = list(
    zip(
        range(-DATA_QUEUE_SIZE, 1, FPS),
        map(str, range(-DATA_QUEUE_SIZE, 1, FPS)),
    )
)
STEP_MINOR_TICKS = list(
    zip(
        range(-DATA_QUEUE_SIZE, 1, FPS // 2),
        ["" for _ in range(DATA_QUEUE_SIZE // 2)],
    )
)
STEP_TICKS = [STEP_MAJOR_TICKS, STEP_MINOR_TICKS]

# TODO: send encoder angle in deg as well
ENCODER_RAW2DEG = 360 / 4096


class Colors:
    PRIMARY = "#202020"
    ON_PRIMARY = "#919090"
    SECONDARY = "#1E1E1E"
    ON_SECONDARY = "#878786"
    FOREGROUND = "#131313"
    ON_FOREGROUND = "#565757"
    ON_FOREGROUND_DIM = "#373737"
    ACCENT = "#1A1A1A"
    ON_ACCENT = "#737473"
    ON_ACCENT_DIM = "#4A4A4A"
    GREEN = "#98FB98"
    ORANGE = "#FFCC99"
    PASTEL_BLUE = "#ADD8E6"
    PASTEL_PURPLE = "#DDA0DD"
    PASTEL_YELLOW = "#FFFFE0"


class PltColors:
    PLT_BG = QColor(0x1A, 0x1A, 0x1A, 255)
    STATS_BG = QColor(0x1E, 0x1E, 0x1E, 255)
    PASTEL_GREEN = QColor(152, 251, 152, 200)
    PASTEL_ORANGE = QColor(255, 204, 153, 200)


class PlotStatsWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Sunken)
        self.setStyleSheet(
            f"""
            background-color: {Colors.ACCENT};
            color: {Colors.ON_ACCENT};
            border-radius: 2px;
            padding: 2px;
        """
        )
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 2, 5, 2)


class LatencyStatsWidget(PlotStatsWidget):
    def __init__(self):
        super().__init__()

        # Left
        self.max_label = QLabel("")
        self.min_label = QLabel("")
        self.mean_label = QLabel("")
        self.std_label = QLabel("")
        self.avg_label = QLabel("")

        # Right
        self.current_label = QLabel("")
        self.current_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.layout.addWidget(self.max_label)
        self.layout.addWidget(self.min_label)
        self.layout.addWidget(self.mean_label)
        self.layout.addWidget(self.std_label)
        self.layout.addStretch(1)
        self.layout.addWidget(self.current_label)
        self.layout.addWidget(self.avg_label)


class LatencyPlotWidget(QWidget):
    def __init__(self, name: str, fps_target: float):
        super().__init__()
        self.fps_target = fps_target
        self.dt_target_ms = 1000 / fps_target

        self.dt_q_ms = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)

        self.dt_ma_window = 10
        self.dt_ma_q = deque(10 * [0], maxlen=self.dt_ma_window)
        self.dt_ma_data = np.zeros(DATA_QUEUE_SIZE - self.dt_ma_window + 1)
        self.dt_ma_kernel = np.ones(self.dt_ma_window) / self.dt_ma_window
        self.dt_ma_xs = PLOT_TIME_STEPS[-(self.dt_ma_data.size) :]

        # Colors
        # TODO: create into a stable color palette and move to a separate place
        self.qcolor_dt = QColor(255, 255, 255, 64)  # off-white
        self.qcolor_dt_moving_avg = QColor(0, 165, 128, 200)  # cyan
        self.qcolor_setpoint = QColor(152, 251, 152, 200)  # pastel green
        self.qcolor_max = QColor(255, 100, 100, 150)  # pastel red
        self.qcolor_tolerance = QColor(152, 251, 152, 100)  # pastel green soft
        self.qcolor_tolerance_fill = QColor(152, 251, 152, 16)  # pastel green soft fill

        self.w_plot = PlotWidget()
        self.w_plot.setBackground(PltColors.PLT_BG.name())
        self.w_plot.setXRange(-DATA_QUEUE_SIZE, 0)
        self.w_plot.getAxis("bottom").setTicks(STEP_TICKS)
        self.w_plot.getPlotItem().showGrid(x=True, y=True, alpha=0.25)
        self.w_plot.getPlotItem().setTitle(name)
        self.w_plot.getPlotItem().setLabel("left", "Time [ms]")
        self.w_plot.getPlotItem().setLabel("bottom", f"Step [n] (s = {FPS} steps)")

        self.w_plot_legend = self.w_plot.addLegend()
        self.w_plot_legend.anchor(itemPos=(0, 0), parentPos=(0, 1), offset=(8, -35))
        self.w_plot_legend.setBrush(QBrush(QColor(40, 40, 40, 255)))
        self.w_plot_legend.setPen(mkPen(color="#555555", width=0.5))
        self.w_plot_legend.layout.setContentsMargins(3, 1, 3, 1)
        self.w_plot_legend.setColumnCount(5)

        self.stats_widget = LatencyStatsWidget()

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.w_plot, stretch=1)
        layout.addWidget(self.stats_widget)

        self._setup_threshold_lines()
        self._setup_dt_moving_average()
        self._setup_primary_plots()

        self.setMinimumHeight(200)

    def _setup_primary_plots(self):
        dt_pen = mkPen(self.qcolor_dt, style=Qt.PenStyle.DotLine)
        self.dt_plot = self.w_plot.plot(PLOT_TIME_STEPS, self.dt_q_ms, pen=dt_pen, name="Current")

    def _setup_dt_moving_average(self):
        mean_pen = mkPen(self.qcolor_dt_moving_avg, style=Qt.DotLine)
        mean_name = "Mean"
        self.mean_line = pg.InfiniteLine(angle=0, pen=mean_pen, movable=False, name=mean_name)
        self.w_plot.addItem(self.mean_line)
        self.w_plot_legend.addItem(PlotCurveItem(pen=mean_pen), name=mean_name)

        self.dt_ma_plot = self.w_plot.plot(
            self.dt_ma_xs,
            self.dt_ma_data,
            pen=mkPen(self.qcolor_dt_moving_avg, style=Qt.SolidLine, width=2),
            name="Avg T (10 steps)",
        )

        self.dt_ma_plot_last = ScatterPlotItem(size=8, pen=None, brush=self.qcolor_dt_moving_avg)
        self.dt_ma_plot_last.setZValue(2)
        self.w_plot.addItem(self.dt_ma_plot_last)

    def _setup_threshold_lines(self):
        tolerance_factor = 0.10

        thresh_pen = mkPen(self.qcolor_setpoint, style=Qt.PenStyle.DashLine, width=1)
        thresh_name = f"Target ±{tolerance_factor * 100}%"
        self.thresh_line_setpoint = pg.InfiniteLine(
            angle=0,
            pen=thresh_pen,
            label=f"{self.dt_target_ms:.1f}ms ({self.fps_target} FPS)",
            labelOpts={"position": 0.11, "color": self.qcolor_setpoint, "movable": True},
        )

        self.thresh_line_setpoint.setValue(self.dt_target_ms)
        self.w_plot.addItem(self.thresh_line_setpoint)
        self.w_plot_legend.addItem(PlotCurveItem(pen=thresh_pen), name=thresh_name)

        self._setup_target_region(tolerance_factor)

        max_pen = mkPen(self.qcolor_max, style=Qt.DotLine)
        max_name = "Max"
        self.max_line = pg.InfiniteLine(angle=0, pen=max_pen, movable=False)
        self.w_plot.addItem(self.max_line)
        self.w_plot_legend.addItem(PlotCurveItem(pen=max_pen), name=max_name)

    def _setup_target_region(self, tolerance_factor: float):
        lower_bound = self.dt_target_ms * (1 - tolerance_factor)
        upper_bound = self.dt_target_ms * (1 + tolerance_factor)

        xs = np.array(PLOT_TIME_STEPS)
        low = pg.PlotCurveItem(x=xs, y=np.ones_like(xs) * lower_bound)
        high = pg.PlotCurveItem(x=xs, y=np.ones_like(xs) * upper_bound)

        self.target_region = pg.FillBetweenItem(low, high, brush=QBrush(self.qcolor_tolerance_fill))
        self.target_region.setZValue(-1)

        tolerance_pen = mkPen(self.qcolor_tolerance, width=1, style=Qt.DotLine)
        self.lower_tolerance_line = pg.InfiniteLine(pos=lower_bound, angle=0, pen=tolerance_pen)
        self.upper_tolerance_line = pg.InfiniteLine(pos=upper_bound, angle=0, pen=tolerance_pen)

        self.w_plot.addItem(self.upper_tolerance_line)
        self.w_plot.addItem(self.target_region)
        self.w_plot.addItem(self.lower_tolerance_line)

    def _update_statistics_display(self, stats):
        current_html = f"<span>Current: {stats['current']:.2f} ms</span>"
        avg_html = f"<span style='color: {self.qcolor_dt_moving_avg.name()}'>Average: {self.dt_ma_data[-1]:.2f} ms</span>"
        mean_html = f"<span style='color: {self.qcolor_dt_moving_avg.name()};'>Mean: {stats['mean']:.2f} ms</span>"
        min_html = f"<span style='color: #AAAAAA;'>Min: {stats['min']:.2f} ms</span>"
        max_html = (
            f"<span style='color: {self.qcolor_max.name()};'>Max: {stats['max']:.2f} ms</span>"
        )
        std_html = f"<span style='color: #AAAAAA;'>Std: {stats['std']:.2f} ms</span>"

        self.stats_widget.current_label.setText(current_html)
        self.stats_widget.avg_label.setText(avg_html)
        self.stats_widget.mean_label.setText(mean_html)
        self.stats_widget.min_label.setText(min_html)
        self.stats_widget.max_label.setText(max_html)
        self.stats_widget.std_label.setText(std_html)

    def update(self, dt_ms: float):
        self.dt_q_ms.append(dt_ms)
        self.dt_ma_q.append(dt_ms)
        stats = self._calculate_statistics()

        self.dt_plot.setData(PLOT_TIME_STEPS, self.dt_q_ms)
        self._update_statistics_display(stats)
        self._update_reference_lines(stats)
        self._update_moving_average()
        self._update_y_scale(stats["max"], stats["min"])

    def _calculate_statistics(self):
        dt_array = np.array(self.dt_q_ms)
        return {
            "mean": np.mean(dt_array),
            "max": np.max(dt_array),
            "min": np.min(dt_array),
            "std": np.std(dt_array),
            "current": self.dt_q_ms[-1],
        }

    def _update_reference_lines(self, stats):
        self.mean_line.setValue(stats["mean"])
        self.max_line.setValue(stats["max"])

    def _update_moving_average(self):
        self.dt_ma_data = np.roll(self.dt_ma_data, -1)
        self.dt_ma_data[-1] = np.sum(np.array(self.dt_ma_q) * self.dt_ma_kernel)

        self.dt_ma_plot.setData(x=self.dt_ma_xs, y=self.dt_ma_data)
        self.dt_ma_plot_last.setData([self.dt_ma_xs[-1]], [self.dt_ma_data[-1]])

    def _update_y_scale(self, current_max: float, current_min: float):
        buffer = 1.1
        current_view_min, current_view_max = self.w_plot.viewRange()[1]

        target_min = max(0, current_min)
        target_max = max(self.dt_target_ms * buffer, current_max * buffer)

        too_big = (target_min < current_view_min) or (current_view_max > (target_max * buffer))
        too_small = target_max > current_view_max

        if too_big or too_small:
            self.w_plot.setYRange(target_min, target_max, padding=0.05)


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

    def update(self, measured_speed: float, target_speed: float, engine_power: float):
        self.plot_widget.update(measured_speed, target_speed, engine_power)
        self.stats_widget.update(measured_speed, target_speed, engine_power)


class SpeedPlotStatsWidget(PlotStatsWidget):
    def __init__(self):
        super().__init__()

        self.texts = {
            "speed": f"<span>Speed: </span><span style='color: {Colors.GREEN}'>{{:2.2f}}</span> m/s",
            "target": f"<span>Target: </span><span style='color: {Colors.GREEN}'>{{:2.2f}}</span> m/s",
            "error": f"<span>Error: </span><span>{{:.2f}}</span> m/s",
            "power": f"<span>Power: </span><span style='color: {Colors.ORANGE};'>{{:3.2f}}</span> %",
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

    def update(self, measured_speed: float, target_speed: float, engine_power: float):
        self.speed_label.setText(self.texts["speed"].format(measured_speed))
        self.target_label.setText(self.texts["target"].format(target_speed))
        self.error_label.setText(self.texts["error"].format(measured_speed - target_speed))
        self.power_label.setText(self.texts["power"].format(engine_power * 100))


class SpeedPlotWidget(PlotWidget):
    def __init__(self):
        super().__init__()
        self.setXRange(-DATA_QUEUE_SIZE, 0)
        self.setBackground(Colors.FOREGROUND)

        # Initialize data queues
        self.dq_speed = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)
        self.dq_set_speed = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)
        self.dq_engine_power = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)

        # Define consistent colors
        self.speed_color = Colors.GREEN
        self.power_color = Colors.ORANGE
        self.text_pen = mkPen(Colors.ON_ACCENT)
        self.mono_font = DefaultMonospaceFont()

        self.getAxis("left").setPen(self.text_pen)
        self.getAxis("left").setTextPen(self.text_pen)

        self.getAxis("bottom").setTicks(STEP_TICKS)
        self.getAxis("bottom").setPen(self.text_pen)
        self.getAxis("bottom").setTextPen(self.text_pen)

        # Set up grid and title
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.5)
        self.getPlotItem().setTitle("Longitudinal Control")

        # self.getPlotItem().setLabel("left", "Speed [m/s]", color=Colors.ON_ACCENT)
        # self.getPlotItem().setLabel("bottom", f"Step [n]")

        self.power_y_axis = pg.AxisItem("right")
        self.power_y_axis.setPen(self.text_pen)
        self.power_y_axis.setTextPen(self.text_pen)
        # self.power_y_axis.setLabel("Power [%]", color=Colors.ON_ACCENT)
        self.getPlotItem().layout.addItem(self.power_y_axis, 2, 3)

        self.power_viewbox = pg.ViewBox()
        self.power_y_axis.linkToView(self.power_viewbox)
        self.power_viewbox.setXLink(self.getPlotItem())
        self.power_viewbox.setRange(yRange=[-100, 100])
        self.getPlotItem().scene().addItem(self.power_viewbox)

        self.legend = self.getPlotItem().addLegend()
        self.legend.anchor(itemPos=(0, 0), parentPos=(0, 1), offset=(15, -35))
        self.legend.setBrush(QBrush(QColor(Colors.ACCENT)))
        self.legend.setPen(mkPen(color=Colors.ON_FOREGROUND, width=0.5))
        self.legend.layout.setContentsMargins(3, 1, 3, 1)
        self.legend.setColumnCount(3)

        self.plt_speed = self.plot(pen=mkPen(self.speed_color, style=Qt.SolidLine), name="Current")
        self.plt_speed.setData(PLOT_TIME_STEPS, self.dq_speed)
        self.plt_set_speed = self.plot(
            pen=mkPen(self.speed_color, style=Qt.DashLine), name="Target"
        )
        self.plt_set_speed.setData(PLOT_TIME_STEPS, self.dq_set_speed)

        # Power plot
        plt_power_pen = mkPen(self.power_color, style=Qt.SolidLine)
        self.plt_power = pg.PlotCurveItem(pen=plt_power_pen, name="Power")
        self.plt_power.setData(PLOT_TIME_STEPS, np.array(self.dq_engine_power))
        self.power_viewbox.addItem(self.plt_power)
        self.legend.addItem(PlotCurveItem(pen=plt_power_pen), "Power")

        # Markers
        self.plt_current_speed = ScatterPlotItem(size=5, pen=None, brush=self.speed_color)
        self.plt_current_speed.setZValue(1)
        self.addItem(self.plt_current_speed)

        self.plt_current_engine_power = ScatterPlotItem(size=5, pen=None, brush=self.power_color)
        self.plt_current_engine_power.setZValue(1)
        self.power_viewbox.addItem(self.plt_current_engine_power)

        # Resize event for power viewbox
        self.getPlotItem().vb.sigResized.connect(self._update_views)

    def _update_views(self):
        # Keep power_viewbox synced with main viewbox on resize
        self.power_viewbox.setGeometry(self.getPlotItem().vb.sceneBoundingRect())

    def update(self, measured_speed: float, target_speed: float, engine_power: float):
        engine_power_percent = engine_power * 100

        self.dq_speed.append(measured_speed)
        self.dq_set_speed.append(target_speed)
        self.dq_engine_power.append(engine_power_percent)

        self.plt_speed.setData(PLOT_TIME_STEPS, self.dq_speed)
        self.plt_set_speed.setData(PLOT_TIME_STEPS, self.dq_set_speed)
        self.plt_power.setData(PLOT_TIME_STEPS, np.array(self.dq_engine_power))

        self.plt_current_speed.setData([PLOT_TIME_STEPS[-1]], [measured_speed])
        self.plt_current_engine_power.setData([PLOT_TIME_STEPS[-1]], [engine_power_percent])

        self._update_y_scales()

    def _update_y_scales(self):
        # Speed
        speed_buffer = 0.1  # 10% margin
        speed_min = min(min(self.dq_speed), min(self.dq_set_speed))
        speed_max = max(max(self.dq_speed), max(self.dq_set_speed))

        # Add margin to values (min can't go below zero)
        speed_range_min = speed_min * (1 - speed_buffer)
        speed_range_max = speed_max * (1 + speed_buffer)

        # Get current view range
        current_speed_min, current_speed_max = self.viewRange()[1]

        # Check if adjustment needed (avoid constant rescaling)
        speed_too_small = speed_range_max > current_speed_max * 0.95
        speed_too_big = (
            current_speed_max > speed_range_max * 1.5 or current_speed_min < speed_range_min * 0.5
        )

        if speed_too_small or speed_too_big:
            self.setYRange(speed_range_min, speed_range_max, padding=0)

        # power_range_min = min(self.dq_engine_power)
        # power_range_max = max(self.dq_engine_power)

        # # Get current power view range
        # current_power_min, current_power_max = self.power_viewbox.viewRange()[1]

        # # Check if adjustment needed
        # power_too_small = power_range_max > current_power_max * 0.75
        # power_too_big = current_power_max > power_range_max * 1.5

        # if power_too_small or power_too_big:
        #     self.power_viewbox.setYRange(power_range_min, power_range_max, padding=0)


class SteeringPlotWidget(PlotWidget):
    def __init__(self):
        super().__init__()
        self.setXRange(-25, 25)
        self.getAxis("left").setTicks(STEP_TICKS)
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.25)
        self.setBackground(Colors.FOREGROUND)
        self.getPlotItem().setTitle("Steering Angle")
        self.getPlotItem().setLabel("left", f"Step [n] (s = {FPS} steps)")
        self.getPlotItem().setLabel("bottom", "Steering angle [deg]")

        self.steering_deque = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)
        self.set_steering_deque = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)

        self.steering_plot = self.plot(
            self.steering_deque,
            PLOT_TIME_STEPS,
            pen=mkPen(QColor(255, 0, 0, 255), style=Qt.SolidLine),
            name="Value",
        )
        self.set_steering_plot = self.plot(
            self.set_steering_deque,
            PLOT_TIME_STEPS,
            pen=mkPen(QColor(255, 0, 0, 128), style=Qt.DashLine),
            name="Setpoint",
        )

        self.steering_marker = ScatterPlotItem(size=5, pen=mkPen(None), brush="w", name="Current")
        self.steering_marker.setZValue(1)
        self.addItem(self.steering_marker)
        self.getPlotItem().addLegend()

    def update(self, steering_deg, set_steering_deg):
        self.steering_deque.append(steering_deg)
        self.steering_plot.setData(self.steering_deque, PLOT_TIME_STEPS)
        self.steering_marker.setData([steering_deg], [PLOT_TIME_STEPS[-1]])
        self.set_steering_deque.append(set_steering_deg)
        self.set_steering_plot.setData(self.set_steering_deque, PLOT_TIME_STEPS)


class MapPlotWidget(PlotWidget):
    DIR_LINE_LEN = 20

    def __init__(self):
        super().__init__()
        self.setBackground(Colors.FOREGROUND)
        self.setAspectLocked(True)
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.25)
        self.getPlotItem().setTitle("Vehicle Position (X, Y, Yaw)")
        self.dir_x = PlotCurveItem(pen=mkPen("r", width=2))
        self.dir_x.setZValue(1)
        self.dir_y = PlotCurveItem(pen=mkPen("g", width=2))
        self.dir_y.setZValue(1)
        self.addItem(self.dir_x)
        self.addItem(self.dir_y)

        self.x_deque = deque(maxlen=DATA_QUEUE_SIZE)
        self.y_deque = deque(maxlen=DATA_QUEUE_SIZE)

        self.path = PlotCurveItem(pen=mkPen(QColor(255, 255, 255, 255), width=2, style=Qt.DashLine))
        gradient = QLinearGradient(0, 0, 0, 1)
        gradient.setColorAt(0, QColor(255, 255, 255, 255))
        gradient.setColorAt(1, QColor(255, 255, 255, 64))
        self.path.setBrush(QBrush(gradient))
        self.addItem(self.path)

    def update(self, x: float, y: float, yaw: float):
        dx = MapPlotWidget.DIR_LINE_LEN * np.cos(np.radians(yaw))
        dy = MapPlotWidget.DIR_LINE_LEN * np.sin(np.radians(yaw))
        self.dir_x.setData([x, x + dx], [y, y + dy])
        self.dir_y.setData([x, x - dy], [y, y + dx])
        self.x_deque.append(x)
        self.y_deque.append(y)
        self.path.setData(list(self.x_deque), list(self.y_deque))


class IMUPlotWidget(GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(self.sizePolicy())
        self.setBackgroundColor(Colors.FOREGROUND)
        self.setCameraPosition(pos=QVector3D(0, 0, 0), distance=10, azimuth=225)

        # Create the arrows using GLLinePlotItem
        self.arrow_x = GLLinePlotItem(
            pos=np.array([[0, 0, 0], [1, 0, 0]]), color=(1, 0, 0, 1), width=3
        )
        self.arrow_y = GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 1, 0]]), color=(0, 1, 0, 1), width=3
        )
        self.arrow_z = GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 1]]), color=(0, 0, 1, 1), width=3
        )
        self.addItem(self.arrow_x)
        self.addItem(self.arrow_y)
        self.addItem(self.arrow_z)

        # Add axis labels using GLLinePlotItem
        x_label = GLLinePlotItem(
            pos=np.array([[1, 0, 0], [1.1, 0, 0]]), color=(1, 0, 0, 1), width=3
        )
        y_label = GLLinePlotItem(
            pos=np.array([[0, 1, 0], [0, 1.1, 0]]), color=(0, 1, 0, 1), width=3
        )
        z_label = GLLinePlotItem(
            pos=np.array([[0, 0, 1], [0, 0, 1.1]]), color=(0, 0, 1, 1), width=3
        )
        self.addItem(x_label)
        self.addItem(y_label)
        self.addItem(z_label)

        grid = GLGridItem()
        grid.setSize(x=10, y=10, z=10)
        self.addItem(grid)

    def update_data(self, rotation: Rotation):
        roll, pitch, yaw = (
            np.deg2rad(rotation.roll),
            np.deg2rad(rotation.pitch),
            np.deg2rad(rotation.yaw),
        )
        # Create rotation matrices
        Rx = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(roll), -np.sin(roll), 0],
                [0, np.sin(roll), np.cos(roll), 0],
                [0, 0, 0, 1],
            ]
        )

        Ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch), 0],
                [0, 1, 0, 0],
                [-np.sin(pitch), 0, np.cos(pitch), 0],
                [0, 0, 0, 1],
            ]
        )

        Rz = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, 0],
                [np.sin(yaw), np.cos(yaw), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Update arrow orientations
        self.arrow_x.setTransform(R)
        self.arrow_y.setTransform(R)
        self.arrow_z.setTransform(R)


class EncodersPlotWidget(PlotWidget):
    def __init__(self):
        super().__init__()
        self.setAspectLocked()
        self.setXRange(-250, 250)
        self.setYRange(-250, 250)
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.25)
        self.setBackground(Colors.SECONDARY)
        self.getPlotItem().setTitle("Encoders Data")

        self.l_deque = deque(maxlen=150)
        self.r_deque = deque(maxlen=150)

        self.l_scatter = ScatterPlotItem(size=5, pen=mkPen(None), brush=mkBrush(255, 0, 0, 120))
        self.addItem(self.l_scatter)

        self.r_scatter = ScatterPlotItem(size=5, pen=mkPen(None), brush=mkBrush(0, 255, 0, 120))
        self.addItem(self.r_scatter)

        self.l_curve = PlotCurveItem(pen=mkPen("r", width=1))
        self.addItem(self.l_curve)

        self.r_curve = PlotCurveItem(pen=mkPen("g", width=1))
        self.addItem(self.r_curve)

    def update(self, left_data: EncoderData, right_data: EncoderData):
        EncodersPlotWidget._update_encoder(left_data, self.l_deque, self.l_scatter, self.l_curve)
        EncodersPlotWidget._update_encoder(right_data, self.r_deque, self.r_scatter, self.r_curve)

    def _update_encoder(
        data: EncoderData, queue: deque, scatter: ScatterPlotItem, curve: PlotCurveItem
    ):
        deg = data.position * ENCODER_RAW2DEG
        rad = np.deg2rad(deg)
        x = data.magnitude * np.cos(rad)
        y = data.magnitude * np.sin(rad)

        queue.append((x, y))
        scatter.setData(queue)
        curve.setData(queue)
