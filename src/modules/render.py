import sys
import time
import cv2
import numpy as np
from collections import deque

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap, QColor, QBrush, QLinearGradient, QVector3D
from PySide6.QtWidgets import QApplication, QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QWidget
import pyqtgraph as pg  # type: ignore
import pyqtgraph.opengl as gl  # type: ignore


from utils.data import JPGImageData
from utils.ipc import SPMCQueue


FPS = 60
DTIME = 1 / FPS
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

# Contrast control (1.0-3.0)
ALPHA = 3
# Brightness control (0-100)
BETA = 10

RAW2DEG = 360 / 4096


class RendererUISetup(QThread):
    ui_setup_data_ready = Signal(tuple)

    def __init__(
        self,
        q_image: SPMCQueue,
        q_sensor: SPMCQueue,
        q_remote: SPMCQueue,
    ):
        super().__init__()
        self._q_image = q_image
        self._q_sensor = q_sensor
        self._q_remote = q_remote

    def run(self):
        q_image = self._q_image.get_consumer()
        jpg_image_data: JPGImageData = q_image.get()
        image_array = cv2.imdecode(np.frombuffer(jpg_image_data.jpg, np.uint8), cv2.IMREAD_COLOR)
        height, width, _ = image_array.shape
        self.ui_setup_data_ready.emit((width, height))


class RendererImageData(QThread):
    image_data_ready = Signal(tuple)

    def __init__(self, q_image: SPMCQueue):
        super().__init__()
        self._q_image = q_image
        self._is_running = True

    def run(self):
        q_image = self._q_image.get_consumer()
        while self._is_running:
            jpg_image_data: JPGImageData = q_image.get()
            self.image_data_ready.emit(
                (self._jpg2qimage(jpg_image_data.jpg), jpg_image_data.timestamp)
            )

    def _jpg2qimage(self, jpg):
        image_array = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        height, width, channel = image_array.shape
        bytes_per_line = channel * width
        return QImage(image_array.data, width, height, bytes_per_line, QImage.Format_BGR888)

    def stop(self):
        self._is_running = False


class RendererSensorData(QThread):
    sensor_data_ready = Signal(tuple)

    def __init__(self, q_sensor: SPMCQueue):
        super().__init__()
        self._q_sensor = q_sensor
        self._is_running = True

    def run(self):
        q_sensor = self._q_sensor.get_consumer()
        while self._is_running:
            data = q_sensor.get()
            self.sensor_data_ready.emit(data)

    def stop(self):
        self._is_running = False


class VehicleRendererApp(QWidget):
    window_init_finished = Signal()

    def __init__(self):
        super().__init__()

    def init_window(self, data):
        self._cam_width, self._cam_height = data

        self._w_init_main_window()
        self._w_init_camera_raw()
        self._w_init_camera_depth()
        self._w_init_speed_plot()
        self._w_init_steering_plot()
        self._w_init_map_plot()
        self._w_init_imu_plot()
        self._w_init_plt_encoders()
        self._w_init_layout()

        self.show()
        self.window_init_finished.emit()

    def _w_init_main_window(self):
        self.setWindowTitle("ToySim UI")
        # self.setWindowFlags(Qt.FramelessWindowHint)
        self._drag_pos = None

    def _w_init_camera_raw(self):
        self.rgb_label = QLabel(self)
        self.rgb_label.setMinimumSize(self._cam_width, self._cam_height)
        self.rgb_pixmap = QPixmap()

    def _w_init_camera_depth(self):
        self.depth_label = QLabel(self)
        # self.depth_label.setMinimumSize(VehicleCamera.WIDTH, VehicleCamera.HEIGHT)
        self.depth_pixmap = QPixmap()

    def _w_init_speed_plot(self):
        self.speed_plot = pg.PlotWidget()
        self.speed_plot.setXRange(-DATA_QUEUE_SIZE, 0)
        self.speed_plot.setYRange(-200, 200)
        self.speed_plot.getAxis("bottom").setTicks(STEP_TICKS)
        self.speed_plot.getPlotItem().showGrid(x=True, y=True)
        self.speed_plot.getPlotItem().setTitle("Speed")
        self.speed_plot.getPlotItem().setLabel("left", "Speed [cm/s]")
        self.speed_plot.getPlotItem().setLabel("bottom", f"Step [n] (s = {FPS} steps)")
        self.speed_marker = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush="w")
        self.speed_marker.setZValue(1)
        self.speed_plot.addItem(self.speed_marker)

        self.speed_data = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)
        self.speed_setpoint_data = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)

        self.speed_plot.getPlotItem().addLegend()
        self.speed_plot_data = self.speed_plot.plot(
            PLOT_TIME_STEPS,
            self.speed_data,
            pen=pg.mkPen(QColor(0, 255, 0, 255), style=Qt.SolidLine),
            name="Value",
        )
        self.speed_setpoint_plot_data = self.speed_plot.plot(
            PLOT_TIME_STEPS,
            self.speed_setpoint_data,
            pen=pg.mkPen(QColor(0, 255, 0, 64), style=Qt.DashLine),
            name="Setpoint",
        )

    def _w_init_steering_plot(self):
        self.steering_plot = pg.PlotWidget()
        # self.steering_plot.setMinimumSize(VehicleCamera.WIDTH, VehicleCamera.HEIGHT)
        self.steering_plot.setXRange(-40, 40)
        self.steering_plot.getAxis("left").setTicks(STEP_TICKS)
        self.steering_plot.getPlotItem().showGrid(x=True, y=True)
        self.steering_plot.getPlotItem().setTitle("Steering Angle")
        self.steering_plot.getPlotItem().setLabel("left", f"Step [n] (s = {FPS} steps)")
        self.steering_plot.getPlotItem().setLabel("bottom", "Steering angle [deg]")
        self.steering_marker = pg.ScatterPlotItem(
            size=5, pen=pg.mkPen(None), brush="w", name="Current"
        )
        self.steering_marker.setZValue(1)
        self.steering_plot.addItem(self.steering_marker)

        self.steering_data = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)
        self.steering_setpoint_data = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)

        self.steering_plot.getPlotItem().addLegend()
        self.steering_plot_data = self.steering_plot.plot(
            self.steering_data,
            PLOT_TIME_STEPS,
            pen=pg.mkPen(QColor(255, 0, 0, 255), style=Qt.SolidLine),
            name="Value",
        )
        self.steering_setpoint_plot_data = self.steering_plot.plot(
            self.steering_setpoint_data,
            PLOT_TIME_STEPS,
            pen=pg.mkPen(QColor(255, 0, 0, 128), style=Qt.DashLine),
            name="Setpoint",
        )

    def _w_init_map_plot(self):
        self.map_plot = pg.PlotWidget()
        self.map_plot.setXRange(-100, 100)  # Adjust the range as needed
        self.map_plot.setYRange(-100, 100)  # Adjust the range as needed
        self.map_plot.getPlotItem().showGrid(x=True, y=True)
        self.map_plot.getPlotItem().setTitle("Vehicle Position (X, Y, Yaw)")
        # self.arrow = pg.ArrowItem(angle=90, tipAngle=30, baseAngle=20, headLen=20, tailLen=10, headWidth=10, tailWidth=4, pen={'color': 'g', 'width': 2})
        # self.map_plot.addItem(self.arrow)
        # self.arrow.setZValue(1)
        self.orientation_line1 = pg.PlotCurveItem(pen=pg.mkPen("r", width=2))
        self.orientation_line1.setZValue(1)
        self.orientation_line2 = pg.PlotCurveItem(pen=pg.mkPen("g", width=2))
        self.orientation_line2.setZValue(1)
        self.map_plot.addItem(self.orientation_line1)
        self.map_plot.addItem(self.orientation_line2)

        # Create a deque to store the past positions
        self.map_plot_positions_x = deque(maxlen=DATA_QUEUE_SIZE)
        self.map_plot_positions_y = deque(maxlen=DATA_QUEUE_SIZE)

        # Create a PlotCurveItem to represent the path
        self.path = pg.PlotCurveItem(
            pen=pg.mkPen(QColor(255, 255, 255, 255), width=2, style=Qt.DashLine)
        )
        gradient = QLinearGradient(0, 0, 0, 1)
        gradient.setColorAt(0, QColor(255, 255, 255, 255))
        gradient.setColorAt(1, QColor(255, 255, 255, 64))
        self.path.setBrush(QBrush(gradient))
        self.map_plot.addItem(self.path)

    def _w_init_imu_plot(self):
        self.imu_plot = gl.GLViewWidget()  # Placeholder for the 3D plot
        self.imu_plot.setSizePolicy(self.map_plot.sizePolicy())
        self.imu_plot.setCameraPosition(pos=QVector3D(0, 0, 0), distance=10, azimuth=225)
        self.imu_plot.setBackgroundColor("k")

        # Create the arrows using GLLinePlotItem
        self.arrow_x = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [1, 0, 0]]), color=(1, 0, 0, 1), width=3
        )
        self.arrow_y = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 1, 0]]), color=(0, 1, 0, 1), width=3
        )
        self.arrow_z = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 1]]), color=(0, 0, 1, 1), width=3
        )

        # Add the arrows to the plot
        self.imu_plot.addItem(self.arrow_x)
        self.imu_plot.addItem(self.arrow_y)
        self.imu_plot.addItem(self.arrow_z)

        grid = gl.GLGridItem()
        grid.setSize(x=10, y=10, z=10)
        self.imu_plot.addItem(grid)

        # Add axis labels using GLLinePlotItem
        x_label = gl.GLLinePlotItem(
            pos=np.array([[1, 0, 0], [1.1, 0, 0]]), color=(1, 0, 0, 1), width=3
        )
        y_label = gl.GLLinePlotItem(
            pos=np.array([[0, 1, 0], [0, 1.1, 0]]), color=(0, 1, 0, 1), width=3
        )
        z_label = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 1], [0, 0, 1.1]]), color=(0, 0, 1, 1), width=3
        )
        self.imu_plot.addItem(x_label)
        self.imu_plot.addItem(y_label)
        self.imu_plot.addItem(z_label)

    def _w_init_plt_encoders(self):
        self.plt_encoders = pg.PlotWidget()
        # self.right_encoder_plot = pg.PlotWidget()
        self.plt_encoders.setAspectLocked()
        self.plt_encoders.setXRange(-250, 250)
        self.plt_encoders.setYRange(-250, 250)
        self.plt_encoders.getPlotItem().showGrid(x=True, y=True)
        self.plt_encoders.getPlotItem().setTitle("Encoders Data")
        self.plt_encoders_left_data = deque(maxlen=150)
        self.plt_encoders_right_data = deque(maxlen=150)
        self.plt_encoders_left_scatter = pg.ScatterPlotItem(
            size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120)
        )
        self.plt_encoders_right_scatter = pg.ScatterPlotItem(
            size=5, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 120)
        )
        self.plt_encoders.addItem(self.plt_encoders_left_scatter)
        self.plt_encoders.addItem(self.plt_encoders_right_scatter)
        self.plt_encoders_left_curve = pg.PlotCurveItem(pen=pg.mkPen("r", width=1))
        self.plt_encoders_right_curve = pg.PlotCurveItem(pen=pg.mkPen("g", width=1))
        self.plt_encoders.addItem(self.plt_encoders_left_curve)
        self.plt_encoders.addItem(self.plt_encoders_right_curve)

    def _w_init_layout(self):
        imu_layout = QHBoxLayout()
        imu_layout.addWidget(self.imu_plot)
        imu_layout.addWidget(self.map_plot)

        encoders_layout = QHBoxLayout()
        encoders_layout.addWidget(self.plt_encoders)
        encoders_layout.addWidget(QLabel("Placeholder for Right Encoder Plot"))

        left_layout = QVBoxLayout()
        left_layout.addLayout(imu_layout)
        left_layout.addLayout(encoders_layout)

        middle_layout = QGridLayout()
        middle_layout.addWidget(self.rgb_label)
        middle_layout.addWidget(self.depth_label)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.speed_plot)
        right_layout.addWidget(self.steering_plot)

        # Create the main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(middle_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)
        self.setStyleSheet("background-color: #2d2a2e;")

    def update_orientation_lines(self, x, y, yaw):
        length = 20  # Length of the orientation lines
        dx = length * np.cos(np.radians(yaw))
        dy = length * np.sin(np.radians(yaw))

        # Line 1: From (x, y) to (x + dx, y + dy)
        self.orientation_line1.setData([x, x + dx], [y, y + dy])

        # Line 2: Perpendicular to Line 1
        self.orientation_line2.setData([x, x - dy], [y, y + dx])

    def update_image_data(self, data):
        qimage, timestamp = data
        self.rgb_pixmap.convertFromImage(qimage)
        self.rgb_label.setPixmap(self.rgb_pixmap)
        # Depth
        # self.depth_pixmap.convertFromImage(qimage_depth)
        # self.depth_label.setPixmap(self.depth_pixmap)

    def update_sensor_data(self, data):
        ...
        # Plot speed
        # speed_cm = vehicle_data.speed / 10
        # self.speed_data.append(speed_cm)
        # self.speed_plot_data.setData(PLOT_TIME_STEPS, self.speed_data)
        # self.speed_marker.setData([PLOT_TIME_STEPS[-1]], [speed_cm])
        # self.speed_setpoint_data.append(control_data.speed_setpoint)
        # self.speed_setpoint_plot_data.setData(PLOT_TIME_STEPS, self.speed_setpoint_data)
        # Plot steering
        # self.steering_data.append(vehicle_data.steering_angle)
        # self.steering_plot_data.setData(self.steering_data, PLOT_TIME_STEPS)
        # self.steering_marker.setData([vehicle_data.steering_angle], [PLOT_TIME_STEPS[-1]])
        # self.steering_setpoint_data.append(control_data.steering_angle_setpoint)
        # self.steering_setpoint_plot_data.setData(self.steering_setpoint_data, PLOT_TIME_STEPS)
        # Arrow
        # self.arrow.setPos(vehicle_data.pose.position.y, vehicle_data.pose.position.x)
        # self.arrow.setStyle(angle=vehicle_data.pose.rotation.yaw+90)
        # Add the new position to the deque
        # self.map_plot_positions_x.append(vehicle_data.pose.position.x)
        # self.map_plot_positions_y.append(vehicle_data.pose.position.y)

        # self.path.setData(list(self.map_plot_positions_x), list(self.map_plot_positions_y))

        # self.update_imu_plot(vehicle_data.pose.rotation)
        self._w_update_plt_encoders(data.rleft_encoder, data.rright_encoder)
        # self.update_orientation_lines(
        #     vehicle_data.pose.position.x,
        #     vehicle_data.pose.position.y,
        #     vehicle_data.pose.rotation.yaw,
        # )

    def update_imu_plot(self, rotation):
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

    def _w_update_plt_encoders(self, left_data, right_data):
        # Convert polar coordinates to Cartesian coordinates
        l_rad = np.deg2rad(left_data.position * RAW2DEG)
        l_x = left_data.magnitude * np.cos(l_rad)
        l_y = left_data.magnitude * np.sin(l_rad)
        r_rad = np.deg2rad(right_data.position * RAW2DEG)
        r_x = right_data.magnitude * np.cos(r_rad)
        r_y = right_data.magnitude * np.sin(r_rad)

        # Append the new data point to the deque
        self.plt_encoders_left_data.append((l_x, l_y))
        self.plt_encoders_right_data.append((r_x, l_y))

        # TODO - try to only add new data without full redraw
        self.plt_encoders_left_scatter.setData(
            [p[0] for p in self.plt_encoders_left_data], [p[1] for p in self.plt_encoders_left_data]
        )
        self.plt_encoders_left_curve.setData(
            [p[0] for p in self.plt_encoders_left_data], [p[1] for p in self.plt_encoders_left_data]
        )
        self.plt_encoders_right_scatter.setData(
            [p[0] for p in self.plt_encoders_right_data], [p[1] for p in self.plt_encoders_right_data]
        )
        self.plt_encoders_right_curve.setData(
            [p[0] for p in self.plt_encoders_right_data], [p[1] for p in self.plt_encoders_right_data]
        )


class Renderer:
    def __init__(
        self,
        q_image: SPMCQueue,
        q_sensor: SPMCQueue,
        q_remote: SPMCQueue,
    ):
        self._q_image = q_image
        self._q_sensor = q_sensor
        self._q_remote = q_remote

    def run(self):
        app = QApplication(sys.argv)
        app.aboutToQuit.connect(self._stop_thread_and_wait)

        ex: QWidget = VehicleRendererApp()

        self._t_ui_setup = RendererUISetup(
            q_image=self._q_image, q_sensor=self._q_sensor, q_remote=self._q_remote
        )
        self._t_ui_setup.ui_setup_data_ready.connect(ex.init_window)
        self._t_ui_setup.start()

        self._t_image_data = RendererImageData(q_image=self._q_image)
        self._t_image_data.image_data_ready.connect(ex.update_image_data)
        # self._t_image_data.finished.connect(app.exit)
        ex.window_init_finished.connect(self._t_image_data.start)

        self._t_sensor_data = RendererSensorData(q_sensor=self._q_sensor)
        self._t_sensor_data.sensor_data_ready.connect(ex.update_sensor_data)
        self._t_sensor_data.start()

        return app.exec()

    def _stop_thread_and_wait(self):
        self._t_image_data.stop()
        self._t_sensor_data.stop()
        self._t_image_data.wait()
        self._t_sensor_data.wait()
