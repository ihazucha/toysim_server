import numpy as np

from collections import deque

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QLinearGradient, QBrush, QVector3D
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame, QVBoxLayout

from pyqtgraph import PlotWidget, ScatterPlotItem, PlotCurveItem, mkPen, mkBrush
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLGridItem

from datalink.data import EncoderData, Rotation

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


class LatencyPlot(QWidget):  # Changed from PlotWidget to QWidget
    def __init__(self, name: str, fps_setpoint: float):
        super().__init__()
        
        # Create the actual plot widget
        self.plot_widget = PlotWidget()
        
        # Create stats display widget
        self.stats_widget = QFrame()
        self.stats_widget.setFrameShape(QFrame.StyledPanel)
        self.stats_widget.setFrameShadow(QFrame.Sunken)
        self.stats_widget.setStyleSheet("""
            background-color: #1e1e1e;
            color: white;
            border-radius: 2px;
            padding: 2px;
        """)
        
        # Create statistics labels
        self.stats_layout = QHBoxLayout(self.stats_widget)
        self.stats_layout.setContentsMargins(5, 2, 5, 2)
        
        # Current value (right-aligned)
        self.current_label = QLabel("")
        self.current_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # Other stats (left-aligned)
        self.mean_label = QLabel("")
        self.min_label = QLabel("")
        self.max_label = QLabel("")
        self.std_label = QLabel("")
        
        # Add labels to stats layout with spacers
        self.stats_layout.addWidget(self.mean_label)
        self.stats_layout.addWidget(self.min_label)
        self.stats_layout.addWidget(self.max_label)
        self.stats_layout.addWidget(self.std_label)
        self.stats_layout.addStretch(1)  # Flexible space
        self.stats_layout.addWidget(self.current_label)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget, stretch=1)  # Plot takes most space
        layout.addWidget(self.stats_widget)  # Stats at bottom
        
        # Basic plot setup
        self.plot_widget.setXRange(-DATA_QUEUE_SIZE, 0)
        self.plot_widget.getAxis("bottom").setTicks(STEP_TICKS)
        self.plot_widget.getPlotItem().showGrid(x=True, y=True, alpha=0.5)
        
        # Configuration values
        self.fps_setpoint = fps_setpoint
        self.thresh_fps_setpoint_ms = 1000 / fps_setpoint
        self.window_size = 10
        
        # Colors
        self.qcolor_dt_plot = QColor(255, 255, 255, 64) # off-white
        self.qcolor_ma_plot = QColor(0, 165, 128, 200) # cyan
        self.qcolor_setpoint = QColor(152, 251, 152, 200) # pastel green
        self.qcolor_max = QColor(255, 100, 100, 150) # pastel red
        self.qcolor_tolerance = QColor(152, 251, 152, 100)  # pastel green soft
        self.qcolor_tolerance_fill = QBrush(QColor(152, 251, 152, 16))  # pastel green soft fill 
        
        # Add a title and labels
        # self.plot_widget.getPlotItem().setTitle(name)
        # self.plot_widget.getPlotItem().setLabel("left", "Time [ms]")
        # self.plot_widget.getPlotItem().setLabel("bottom", f"Step [n] (s = {FPS} steps)")

        # Create horizontal legend
        legend = self.plot_widget.addLegend(offset=(10, 10))
        legend.setColumnCount(4)
        legend.addItem(PlotCurveItem(pen=mkPen(self.qcolor_setpoint, width=2)), "Setpoint")
        legend.addItem(PlotCurveItem(pen=mkPen(self.qcolor_max, style=Qt.DotLine)), "Max")

        self.thresholds = {"60fps": 1000 / 60.0, "30fps": 1000 / 30.0, "15fps": 1000 / 15.0}
        
        # Data storage
        self.dt_ms_q = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)
        self.ma_values = np.zeros(DATA_QUEUE_SIZE - self.window_size + 1)
        
        self._setup_primary_plots()
        self._setup_moving_average()
        self._setup_threshold_lines()
        
        # Set initial Y range
        self.plot_widget.setYRange(0, self.thresh_fps_setpoint_ms * 1.5)
        self.setMinimumHeight(200)

    def _setup_primary_plots(self):
        self.dt_plot = self.plot_widget.plot(
            PLOT_TIME_STEPS,
            self.dt_ms_q,
            pen=mkPen(self.qcolor_dt_plot, style=Qt.SolidLine),
            name="Current",
        )
        self.dt_plot_last_value = ScatterPlotItem(size=8, pen=mkPen(None), brush=self.qcolor_dt_plot)
        self.dt_plot_last_value.setZValue(2)
        self.plot_widget.addItem(self.dt_plot_last_value)

    # Update other setup methods to use self.plot_widget instead of self
    def _setup_moving_average(self):
        self.ma_plot = self.plot_widget.plot(
            PLOT_TIME_STEPS[-(len(self.ma_values)) :],
            self.ma_values,
            pen=mkPen(self.qcolor_ma_plot, style=Qt.SolidLine, width=2),
            name="Avg (10pt)",
        )
        self.ma_plot_last_value = ScatterPlotItem(size=8, pen=mkPen(None), brush=self.qcolor_ma_plot)
        self.ma_plot_last_value.setZValue(2)
        self.plot_widget.addItem(self.ma_plot_last_value)
        
        self.mean_line = pg.InfiniteLine(
            angle=0,
            pen=mkPen(self.qcolor_ma_plot, style=Qt.DotLine),
            movable=False,
            label=f"Mean",
            labelOpts={"position": 0.98, "color": self.qcolor_ma_plot, "movable": True},
        )
        self.plot_widget.addItem(self.mean_line)

    def _setup_threshold_lines(self):
        tolerance_factor = 0.10  # 10% tolerance
        lower_bound = self.thresh_fps_setpoint_ms * (1 - tolerance_factor)
        upper_bound = self.thresh_fps_setpoint_ms * (1 + tolerance_factor)

        self.thresh_line_setpoint = pg.InfiniteLine(
            angle=0,
            pen=mkPen(self.qcolor_setpoint, width=2),
            movable=False,
            label=f"{self.thresh_fps_setpoint_ms:.1f}ms ({self.fps_setpoint} FPS) ±{tolerance_factor * 100}%",
            labelOpts={"position": 0.11, "color": self.qcolor_setpoint, "movable": True},
        )
        self.thresh_line_setpoint.setValue(self.thresh_fps_setpoint_ms)
        self.plot_widget.addItem(self.thresh_line_setpoint)

        # Create target region
        self._setup_target_region(lower_bound, upper_bound)

        # Max value indicator
        self.max_line = pg.InfiniteLine(
            angle=0,
            pen=mkPen(self.qcolor_max, style=Qt.DotLine),
            movable=False,
            label=f"Max",
            labelOpts={"position": 0.98, "color": self.qcolor_max, "movable": True},
        )
        self.plot_widget.addItem(self.max_line)

    def _setup_target_region(self, lower_bound, upper_bound):
        """Setup the target region with tolerance bounds"""

        # Create fill region
        x_vals = np.array(PLOT_TIME_STEPS)
        lower_curve = pg.PlotCurveItem(x=x_vals, y=np.ones_like(x_vals) * lower_bound)
        upper_curve = pg.PlotCurveItem(x=x_vals, y=np.ones_like(x_vals) * upper_bound)
        self.target_region = pg.FillBetweenItem(
            lower_curve, upper_curve, brush=self.qcolor_tolerance_fill
        )
        self.target_region.setZValue(-1)
        self.plot_widget.addItem(self.target_region)

        # Add tolerance boundary lines
        self.lower_tolerance_line = pg.InfiniteLine(
            angle=0,
            pen=mkPen(self.qcolor_tolerance, width=1, style=Qt.DotLine),
            movable=False,
        )
        self.lower_tolerance_line.setValue(lower_bound)
        self.plot_widget.addItem(self.lower_tolerance_line)

        self.upper_tolerance_line = pg.InfiniteLine(
            angle=0,
            pen=mkPen(self.qcolor_tolerance, width=1, style=Qt.DotLine),
            movable=False,
        )
        self.upper_tolerance_line.setValue(upper_bound)
        self.plot_widget.addItem(self.upper_tolerance_line)

    def _update_statistics_display(self, stats):
        """Update the statistics display in the separate widget"""
        # Format the values with custom colors
        current_html = f"<span style='color: white; font-weight: bold;'>Current: </span><span style='color: #DDDDDD'>{stats['current']:.2f} ms</span>"
        mean_html = f"<span style='color: {self.qcolor_ma_plot.name()};'>Mean: {stats['mean']:.2f} ms</span>"
        min_html = f"<span style='color: #AAAAAA;'>Min: {stats['min']:.2f} ms</span>"
        max_html = f"<span style='color: {self.qcolor_max.name()};'>Max: {stats['max']:.2f} ms</span>"
        std_html = f"<span style='color: #AAAAAA;'>Std: {stats['std']:.2f} ms</span>"
        
        # Update labels
        self.current_label.setText(current_html)
        self.mean_label.setText(mean_html)
        self.min_label.setText(min_html)
        self.max_label.setText(max_html)
        self.std_label.setText(std_html)

    def update(self, dt_ms: float):
        """Update plot with new latency value"""
        self.dt_ms_q.append(dt_ms)

        self.dt_plot.setData(PLOT_TIME_STEPS, self.dt_ms_q)
        self.dt_plot_last_value.setData([PLOT_TIME_STEPS[-1]], [dt_ms])

        stats = self._calculate_statistics()

        self._update_statistics_display(stats)
        self._update_reference_lines(stats)

        self._update_moving_average()
        self._update_y_scale(stats["max"], stats["min"])

    def _calculate_statistics(self):
        """Calculate statistics from the data queue"""
        dt_array = np.array(self.dt_ms_q)
        return {
            "mean": np.mean(dt_array),
            "max": np.max(dt_array),
            "min": np.min(dt_array),
            "std": np.std(dt_array),
            "current": self.dt_ms_q[-1],
        }

    def _update_reference_lines(self, stats):
        """Update reference lines based on statistics"""
        self.mean_line.setValue(stats["mean"])
        self.max_line.setValue(stats["max"])

    def _update_moving_average(self):
        """Calculate and update the moving average line"""
        if len(self.dt_ms_q) >= self.window_size:
            self.ma_values = np.convolve(
                np.array(self.dt_ms_q), np.ones(self.window_size) / self.window_size, mode="valid"
            )
            ma_x = PLOT_TIME_STEPS[-(self.ma_values.size) :]
            self.ma_plot.setData(ma_x, self.ma_values)
            self.ma_plot_last_value.setData([ma_x[-1]], [self.ma_values[-1]])

    def _update_y_scale(self, current_max: float, current_min: float):
        """Auto-scale the y-axis based on current min/max values"""
        # Add some margin to prevent constant resizing
        buffer_factor = 1.2
        margin = 5  # minimum margin in ms
        
        # Get current view range for comparison
        current_view_min, current_view_max = self.plot_widget.viewRange()[1]
        
        # Calculate target range with buffer
        target_min = max(0, current_min - margin)  # Don't go below zero for latency
        target_max = max(self.thresh_fps_setpoint_ms * buffer_factor, current_max * buffer_factor)
        
        # Only update if there's a significant change to avoid constant rescaling
        if (target_min < current_view_min or 
            target_max > current_view_max or
            current_view_max > target_max * buffer_factor ):  # Shrink view if way too large
            self.plot_widget.setYRange(target_min, target_max, padding=0.05)


class SpeedPlotWidget(PlotWidget):
    def __init__(self):
        super().__init__()
        self.setXRange(-DATA_QUEUE_SIZE, 0)
        self.setYRange(-200, 200)
        self.getAxis("bottom").setTicks(STEP_TICKS)
        self.getPlotItem().showGrid(x=True, y=True)
        self.getPlotItem().setTitle("Speed")
        self.getPlotItem().setLabel("left", "Speed [cm/s]")
        self.getPlotItem().setLabel("bottom", f"Step [n] (s = {FPS} steps)")

        self.speed_marker = ScatterPlotItem(size=5, pen=mkPen(None), brush="w")
        self.speed_marker.setZValue(1)
        self.addItem(self.speed_marker)
        self.getPlotItem().addLegend()

        self.speed_deque = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)
        self.set_speed_deque = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)

        self.speed_plot = self.plot(
            PLOT_TIME_STEPS,
            self.speed_deque,
            pen=mkPen(QColor(0, 255, 0, 255), style=Qt.SolidLine),
            name="Value",
        )
        self.set_speed_plot = self.plot(
            PLOT_TIME_STEPS,
            self.set_speed_deque,
            pen=mkPen(QColor(0, 255, 0, 64), style=Qt.DashLine),
            name="Setpoint",
        )

    def update(self, speed_cmps: float, set_speed_cmps: float):
        """Values in cm/s"""
        self.speed_deque.append(speed_cmps)
        self.speed_plot.setData(PLOT_TIME_STEPS, self.speed_deque)
        self.speed_marker.setData([PLOT_TIME_STEPS[-1]], [speed_cmps])
        self.set_speed_deque.append(set_speed_cmps)
        self.set_speed_plot.setData(PLOT_TIME_STEPS, self.set_speed_deque)


class SteeringPlotWidget(PlotWidget):
    def __init__(self):
        super().__init__()
        self.setXRange(-40, 40)
        self.getAxis("left").setTicks(STEP_TICKS)
        self.getPlotItem().showGrid(x=True, y=True)
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
        self.setXRange(-100, 100)
        self.setYRange(-100, 100)
        self.getPlotItem().showGrid(x=True, y=True)
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
        self.setCameraPosition(pos=QVector3D(0, 0, 0), distance=10, azimuth=225)
        self.setBackgroundColor("k")

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
        self.getPlotItem().showGrid(x=True, y=True)
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
