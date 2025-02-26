import numpy as np

from collections import deque

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QLinearGradient, QBrush, QVector3D

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


class LatencyPlot(PlotWidget):
    def __init__(self, name: str):
        super().__init__()
        self.setXRange(-DATA_QUEUE_SIZE, 0)
        self.getAxis("bottom").setTicks(STEP_TICKS)
        self.getPlotItem().showGrid(x=True, y=True)
        self.getPlotItem().setTitle(f"{name} dt")
        self.getPlotItem().setLabel("left", "dt [ms]")
        self.getPlotItem().setLabel("bottom", f"steps [n]")
        
        self.thresh_60fps = 1000/60.0
        self.thresh_30fps = 1000/30.0
        self.thresh_15fps = 1000/15.0
        
        # Data queue for latency values
        self.dt_q = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)
        
        # Add legend
        self.getPlotItem().addLegend()
        
        # Main latency curve
        self.dt_plot = self.plot(
            PLOT_TIME_STEPS,
            self.dt_q,
            pen=mkPen(QColor(0, 255, 0, 255), style=Qt.SolidLine),
            name="Current",
        )
        
        # Moving average curve
        self.window_size = 10  # 10-point moving average
        self.ma_values = np.zeros(DATA_QUEUE_SIZE - self.window_size + 1)
        self.ma_plot = self.plot(
            PLOT_TIME_STEPS[-(len(self.ma_values)):],
            self.ma_values,
            pen=mkPen(QColor(255, 165, 0, 200), style=Qt.SolidLine, width=2),
            name="Avg (10pt)"
        )
        
        # Current value highlight
        self.dt_plot_last_value = ScatterPlotItem(size=8, pen=mkPen(None), brush="w")
        self.dt_plot_last_value.setZValue(2)
        self.addItem(self.dt_plot_last_value)
        
        # Performance threshold lines
        self.thresh_line_60fps = pg.InfiniteLine(
            angle=0, 
            pen=mkPen('g', width=1.5, style=Qt.DashLine), 
            movable=False, 
            label=f"{self.thresh_60fps:.1f}ms (60 FPS)",
            labelOpts={'position': 0.95, 'color': (255, 255, 0), 'movable': True}
        )
        self.thresh_line_60fps.setValue(self.thresh_60fps)
        self.addItem(self.thresh_line_60fps)
        
        self.thresh_line_30fps = pg.InfiniteLine(
            angle=0, 
            pen=mkPen('y', width=1.5, style=Qt.DashLine), 
            movable=False,
            label=f"{self.thresh_30fps:.1f}ms (30 FPS)",
            labelOpts={'position': 0.95, 'color': (255, 0, 0), 'movable': True}
        )
        self.thresh_line_30fps.setValue(self.thresh_30fps)
        self.addItem(self.thresh_line_30fps)

        self.thresh_line_15fps = pg.InfiniteLine(
            angle=0, 
            pen=mkPen('r', width=1.5, style=Qt.DashLine), 
            movable=False,
            label=f"{self.thresh_15fps:.1f}ms (15 FPS)",
            labelOpts={'position': 0.95, 'color': (255, 0, 0), 'movable': True}
        )
        self.thresh_line_15fps.setValue(self.thresh_15fps)
        self.addItem(self.thresh_line_15fps)
        
        # Reference lines for statistics
        self.mean_line = pg.InfiniteLine(
            angle=0, 
            pen=mkPen(QColor(255, 255, 255, 150), style=Qt.DotLine), 
            movable=False
        )
        self.addItem(self.mean_line)
        
        self.max_line = pg.InfiniteLine(
            angle=0, 
            pen=mkPen(QColor(255, 100, 100, 150), style=Qt.DotLine), 
            movable=False
        )
        self.addItem(self.max_line)
        
        # Statistics text display
        self.stats_text = pg.TextItem(
            html="", 
            anchor=(0, 0),
            fill=QColor(20, 20, 20, 120)
        )
        self.stats_text.setPos(-DATA_QUEUE_SIZE + 5, 10)
        self.addItem(self.stats_text)
        
        # Auto-scaling
        self.auto_scale = True
        self.auto_scale_buffer = 1.2  # 20% buffer
        self.setMinimumHeight(150)
        
        # Set initial Y range
        self.setYRange(0, self.thresh_15fps * 1.5)
        
        # Initialize histogram as a separate widget
        self.histogram = pg.PlotWidget()
        self.histogram.setMaximumHeight(120)
        self.histogram.setLabel('bottom', 'dt [ms]')
        self.histogram.setLabel('left', 'Count')
        self.histogram.showGrid(x=True, y=True)
        self.histogram_plot = self.histogram.plot(
            pen=None,
            stepMode='center',
            fillLevel=0,
            fillBrush=QBrush(QColor(0, 255, 255, 120))
        )

    def update(self, dt: float):
        # Update the data
        self.dt_q.append(dt)
        
        # Update main plot
        self.dt_plot.setData(PLOT_TIME_STEPS, self.dt_q)
        
        # Update current point highlight with color based on performance
        if dt > self.thresh_15fps:
            highlight_color = QColor(255, 0, 0, 200)  # Red for critical
        elif dt > self.thresh_60fps:
            highlight_color = QColor(255, 255, 0, 200)  # Yellow for warning
        else:
            highlight_color = QColor(0, 255, 0, 200)  # Green for good
        
        self.dt_plot_last_value.setBrush(highlight_color)
        self.dt_plot_last_value.setData([PLOT_TIME_STEPS[-1]], [dt])
        
        # Calculate statistics
        dt_array = np.array(self.dt_q)
        mean_dt = np.mean(dt_array)
        max_dt = np.max(dt_array)
        min_dt = np.min(dt_array)
        std_dt = np.std(dt_array)
        
        # Update statistics lines
        self.mean_line.setValue(mean_dt)
        self.max_line.setValue(max_dt)
        
        # Calculate moving average
        if len(self.dt_q) >= self.window_size:
            self.ma_values = np.convolve(
                dt_array, 
                np.ones(self.window_size)/self.window_size, 
                mode='valid'
            )
            ma_x = PLOT_TIME_STEPS[-(len(self.ma_values)):]
            self.ma_plot.setData(ma_x, self.ma_values)
        
        # Update statistics text
        # Determine performance category using all thresholds
        if dt <= self.thresh_60fps:
            current_perf = "60"
        elif dt <= self.thresh_30fps:
            current_perf = "30"
        elif dt <= self.thresh_15fps:
            current_perf = "15"
        else:
            current_perf = "15"
        color_map = {"60": "green", "30": "yellow", "15": "red"}
        
        self.stats_text.setHtml(
            f"""
            <div style='background-color: rgba(20, 20, 20, 120); padding: 0px; border-radius: 3px;'>
                <table style='font-family: monospace; border-collapse: collapse; width: 100%;'>
                    <tr style='font-weight: bold;'>
                        <td>Current:&nbsp;</td>
                        <td style='color: {color_map[current_perf]}'>{dt:.2f}ms</td>
                    </tr>
                    <tr style='color: white;'>
                        <td>Mean</td>
                        <td>{mean_dt:.2f}ms</td>
                    </tr>
                    <tr style='color: white;'>
                        <td>Max</td>
                        <td>{max_dt:.2f}ms</td>
                    </tr>
                    <tr style='color: white;'>
                        <td>Min</td>
                        <td>{min_dt:.2f}ms</td>
                    </tr>
                    <tr style='color: white;'>
                        <td>Std</td>
                        <td>{std_dt:.2f}ms</td>
                    </tr>
                </table>
            </div>
            """
        )
        
        # Auto-scale Y-axis if enabled and needed
        if self.auto_scale:
            max_needed = max(max_dt, self.thresh_15fps) * self.auto_scale_buffer
            current_min, current_max = self.viewRange()[1]
            if max_needed > current_max or max_needed < current_max * 0.7:
                self.setYRange(0, max_needed)
        
        # Update histogram
        y, x = np.histogram(self.dt_q, bins=np.linspace(0, max(max_dt * 1.2, self.thresh_15fps * 1.2), 40))
        self.histogram_plot.setData(x, y)
    
    def toggle_auto_scale(self):
        self.auto_scale = not self.auto_scale
        return self.auto_scale
    
    def get_histogram_widget(self):
        """Returns the histogram widget for placement in layouts"""
        return self.histogram


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
