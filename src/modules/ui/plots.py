import numpy as np

from collections import deque

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QLinearGradient, QBrush, QVector3D

from pyqtgraph import PlotWidget, ScatterPlotItem, PlotCurveItem, mkPen, mkBrush
from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLGridItem

from utils.data import EncoderData, Rotation

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
