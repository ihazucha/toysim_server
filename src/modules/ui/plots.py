
import numpy as np
from collections import deque

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
            "speed": f"<span>Speed: </span><span style='color: {Colors.GREEN}; font-family: monospace;'>{{:+7.2f}}</span> m/s",
            "target": f"<span>Target: </span><span style='color: {Colors.GREEN}; font-family: monospace;'>{{:+7.2f}}</span> m/s",
            "error": f"<span>Error: </span><span style='font-family: monospace;'>{{:+7.2f}}</span> m/s",
            "power": f"<span>Power: </span><span style='color: {Colors.ORANGE}; font-family: monospace;'>{{:+7.2f}}</span> %",
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


# class EncoderPlotWidget(PlotWidget):
#     def __init__(self, name: str):
#         super().__init__()
#         self.setBackground(Colors.FOREGROUND)
#         self.setAspectLocked()
#         # self.setXRange(-100, 100)
#         # self.setYRange(-100, 100)
#         self.getPlotItem().showGrid(x=True, y=True, alpha=0.5)
#         self.getPlotItem().setTitle(name)

#         self.text_pen = mkPen(Colors.ON_ACCENT)
#         self.getAxis("left").setPen(self.text_pen)
#         self.getAxis("left").setTextPen(self.text_pen)
#         self.getAxis("bottom").setPen(self.text_pen)
#         self.getAxis("bottom").setTextPen(self.text_pen)

#         self.plt_scatter = ScatterPlotItem(size=5, pen=mkPen(None), brush=mkBrush(QColor(Colors.ON_FOREGROUND)))
#         self.addItem(self.plt_scatter)

#         # self.plt_curve = PlotCurveItem(pen=mkPen(QColor(Colors.ON_FOREGROUND), width=1))
#         # self.addItem(self.plt_curve)

#         self.dq = deque(maxlen=150)

#     def update(self, data: EncoderData):
#         deg = data.position * ENCODER_RAW2DEG
#         rad = np.deg2rad(deg)
#         x = data.magnitude * np.cos(rad)
#         y = data.magnitude * np.sin(rad)

#         self.dq.append((x, y))
#         # Extract separate x and y arrays from the deque
#         x_values = [point[0] for point in self.dq]
#         y_values = [point[1] for point in self.dq]

#         # Pass separate x and y arrays
#         self.plt_scatter.setData(x=x_values, y=y_values)
#         # self.plt_curve.setData(x=x_values, y=y_values)


class EncoderPlotWidget(QWidget):
    def __init__(self, name: str):
        super().__init__()
        self.main_layout = QVBoxLayout(self)

        self.plot_container = QWidget()
        self.plot_layout = QHBoxLayout(self.plot_container)

        self.polar_plot = PlotWidget()
        self.polar_plot.setBackground(Colors.FOREGROUND)
        self.polar_plot.setAspectLocked(True)
        self.polar_plot.showGrid(x=True, y=True, alpha=0.5)
        self.polar_plot.setTitle(f"{name} - Position")

        # Add plots to layout
        self.plot_layout.addWidget(self.polar_plot, 1)

        # Add separator line
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Sunken)
        self.separator.setStyleSheet(f"background-color: {Colors.ON_FOREGROUND_DIM};")

        # Setup stats display
        self.stats_widget = QFrame()
        self.stats_widget.setStyleSheet(
            f"""
            background-color: {Colors.ACCENT};
            color: {Colors.ON_ACCENT};
            border-radius: 2px;
            padding: 2px;
        """
        )
        self.stats_layout = QHBoxLayout(self.stats_widget)
        self.stats_layout.setContentsMargins(5, 2, 5, 2)

        # Create stats labels
        self.mag_avg_label = QLabel("Mag Avg: 0.00")
        self.mag_std_label = QLabel("Std: 0.00")
        self.angle_stability_label = QLabel("Angle Std: 0.00°")
        self.pos_label = QLabel("Pos: 0°")
        self.pos_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # Add stats to layout
        self.stats_layout.addWidget(self.mag_avg_label)
        self.stats_layout.addWidget(self.mag_std_label)
        self.stats_layout.addWidget(self.angle_stability_label)
        self.stats_layout.addStretch(1)
        self.stats_layout.addWidget(self.pos_label)

        # Add widgets to main layout
        self.main_layout.addWidget(self.plot_container, stretch=1)
        self.main_layout.addWidget(self.separator)
        self.main_layout.addWidget(self.stats_widget)

        # Initialize plot items
        self.text_pen = mkPen(Colors.ON_ACCENT)
        self.configure_axes(self.polar_plot)

        # Circle for reference
        self.circle = pg.ScatterPlotItem(size=1, pen=mkPen(Colors.ON_FOREGROUND_DIM), brush=None)
        initial_circle_radius = 35
        self.draw_reference_circle(initial_circle_radius)
        self.polar_plot.addItem(self.circle)

        # Grid lines (radial)
        for angle in range(0, 360, 30):
            rad = np.deg2rad(angle)
            x = [0, initial_circle_radius * np.cos(rad)]
            y = [0, initial_circle_radius * np.sin(rad)]
            line = pg.PlotCurveItem(
                x=x, y=y, pen=mkPen(Colors.ON_FOREGROUND_DIM, width=0.5, style=Qt.DotLine)
            )
            self.polar_plot.addItem(line)

            # Add angle labels
            text = pg.TextItem(str(angle), anchor=(0.5, 0.5), color=Colors.ON_ACCENT)
            text_offset = initial_circle_radius + 15
            text.setPos(text_offset * np.cos(rad), text_offset * np.sin(rad))
            self.polar_plot.addItem(text)

        # Current point highlight
        self.current_point = ScatterPlotItem(
            size=10, pen=mkPen(Colors.GREEN, width=2), brush=mkBrush(Colors.FOREGROUND)
        )
        self.polar_plot.addItem(self.current_point)

        # Points with color gradient
        self.points = ScatterPlotItem(size=5, pen=None)
        self.polar_plot.addItem(self.points)

        # Data storage
        self.max_samples = 50
        self.angles = np.zeros(self.max_samples)
        self.magnitudes = np.zeros(self.max_samples)
        self.x_values = np.zeros(self.max_samples)
        self.y_values = np.zeros(self.max_samples)
        self.sample_indices = np.arange(self.max_samples)
        self.next_idx = 0
        self.data_count = 0

        # Pre-compute color brushes for efficiency
        self.color_brushes = [
            pg.mkBrush(0, 100, 200, int(255 * i / self.max_samples))
            for i in range(self.max_samples)
        ]

        # Track update counter for throttling non-critical updates
        self.update_counter = 0

        # Pre-compute angle positions for the polar plot
        theta = np.linspace(0, 2 * np.pi, 100)
        self.circle_x = np.cos(theta)
        self.circle_y = np.sin(theta)

        # Initialize rolling stats
        self.rolling_sum = 0
        self.rolling_sum_sq = 0

    def configure_axes(self, plot):
        """Common axis styling for all plots"""
        plot.getAxis("left").setPen(self.text_pen)
        plot.getAxis("left").setTextPen(self.text_pen)
        plot.getAxis("bottom").setPen(self.text_pen)
        plot.getAxis("bottom").setTextPen(self.text_pen)

    def draw_reference_circle(self, radius):
        """Draw reference circle with given radius"""
        theta = np.linspace(0, 2 * np.pi, 100)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        self.circle.setData(x=x, y=y)

    def update(self, data_items):
        """Update widget with multiple encoder data items

        Args:
            data_items: Iterable of EncoderData objects
        """
        # Skip if empty data
        if not data_items:
            return

        # Process all data items in batch
        for data in data_items:
            # Convert encoder data
            deg = data.position * ENCODER_RAW2DEG
            rad = np.deg2rad(deg)
            x = data.magnitude * np.cos(rad)
            y = data.magnitude * np.sin(rad)

            # Update rolling stats
            if self.data_count >= self.max_samples:
                # Remove oldest value from stats
                old_mag = self.magnitudes[self.next_idx]
                self.rolling_sum -= old_mag
                self.rolling_sum_sq -= old_mag * old_mag

            # Add new value to stats
            self.rolling_sum += data.magnitude
            self.rolling_sum_sq += data.magnitude * data.magnitude

            # Store data in circular buffer
            self.angles[self.next_idx] = deg
            self.magnitudes[self.next_idx] = data.magnitude
            self.x_values[self.next_idx] = x
            self.y_values[self.next_idx] = y

            # Update indices
            self.next_idx = (self.next_idx + 1) % self.max_samples
            self.data_count = min(self.data_count + 1, self.max_samples)

        # Always update critical elements - use the last data point for current position
        self.current_point.setData([x], [y])
        self.pos_label.setText(f"Pos: {deg:.1f}°")

        # Throttle remaining updates to reduce CPU usage
        self.update_counter += 1
        if self.update_counter % 3 != 0:
            return

        # Calculate indices efficiently
        if self.data_count < self.max_samples:
            valid_indices = slice(0, self.data_count)
            x_plot = self.x_values[valid_indices]
            y_plot = self.y_values[valid_indices]
            mags = self.magnitudes[valid_indices]
            angs = self.angles[valid_indices]
        else:
            # When buffer is full, we need to handle the wrap-around
            x_plot = np.empty(self.max_samples)
            y_plot = np.empty(self.max_samples)
            mags = np.empty(self.max_samples)
            angs = np.empty(self.max_samples)

            # Copy first segment (from next_idx to end)
            segment1_len = self.max_samples - self.next_idx
            if segment1_len > 0:
                x_plot[:segment1_len] = self.x_values[self.next_idx :]
                y_plot[:segment1_len] = self.y_values[self.next_idx :]
                mags[:segment1_len] = self.magnitudes[self.next_idx :]
                angs[:segment1_len] = self.angles[self.next_idx :]

            # Copy second segment (from start to next_idx)
            if self.next_idx > 0:
                x_plot[segment1_len:] = self.x_values[: self.next_idx]
                y_plot[segment1_len:] = self.y_values[: self.next_idx]
                mags[segment1_len:] = self.magnitudes[: self.next_idx]
                angs[segment1_len:] = self.angles[: self.next_idx]

        # Update path line and points (less critical)
        self.points.setData(x=x_plot, y=y_plot, brush=self.color_brushes[: len(x_plot)])

        # Handle angle wraparound vectorized
        if len(angs) > 1:
            plot_angles = angs.copy()
            # Convert to complex numbers for efficient circular distance calculation
            z = np.exp(1j * np.deg2rad(plot_angles))
            # Calculate phase differences
            diffs = np.angle(z[1:] / z[:-1])
            # Find jumps and apply corrections
            for i in np.where(np.abs(diffs) > np.pi)[0]:
                if diffs[i] > 0:
                    plot_angles[i + 1 :] -= 360
                else:
                    plot_angles[i + 1 :] += 360

        # Calculate statistics efficiently using rolling values
        if self.data_count >= 3:
            mag_avg = self.rolling_sum / self.data_count
            mag_var = (self.rolling_sum_sq / self.data_count) - (mag_avg * mag_avg)
            mag_std = np.sqrt(max(0, mag_var))

            # Only calculate angular std periodically
            if self.update_counter % 10 == 0:
                angle_std = self.calculate_angular_std(angs)
                self.angle_stability_label.setText(f"Angle Std: {angle_std:.2f}°")

            self.mag_avg_label.setText(f"Mag Avg: {mag_avg:.2f}")
            self.mag_std_label.setText(f"Std: {mag_std:.2f}")

        # Auto scale occasionally
        if self.update_counter % 20 == 0:
            self._update_scales()

    def calculate_angular_std(self, angles):
        """Calculate standard deviation for circular data"""
        # Convert to radians
        rad_angles = np.deg2rad(angles)

        # Convert to unit vectors
        x = np.cos(rad_angles)
        y = np.sin(rad_angles)

        # Calculate mean direction
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Calculate circular standard deviation
        r = np.sqrt(x_mean**2 + y_mean**2)
        circular_std = np.sqrt(-2 * np.log(r))

        # Convert back to degrees
        return np.rad2deg(circular_std)

    def _update_scales(self):
        """Update plot scales based on data - optimized version"""
        if self.data_count < 3:
            return

        if self.data_count < self.max_samples:
            max_magnitude = np.max(self.magnitudes[: self.data_count]) * 1.1
        else:
            max_magnitude = np.max(self.magnitudes) * 1.1

        # Update polar plot
        self.polar_plot.setXRange(-max_magnitude, max_magnitude)
        self.polar_plot.setYRange(-max_magnitude, max_magnitude)

        # Update reference circle - reuse pre-computed sin/cos values
        self.circle.setData(x=self.circle_x * max_magnitude, y=self.circle_y * max_magnitude)
