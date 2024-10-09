import sys
import time
import cv2
import numpy as np
from collections import deque
from queue import Queue, Empty

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QLinearGradient, QVector3D
from PySide6.QtWidgets import QApplication, QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QWidget
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from .settings import ClientTypes, SimulationCameraSettings, VehicleCamera
from .processor import SimulationDataFrame, ControlDataFrame, VehicleDataFrame

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


class RendererDataThread(QThread):
    data_ready = Signal(tuple)

    def __init__(self, data_queue: Queue, client, fps=FPS):
        super().__init__()
        self._data_queue = data_queue
        self._client = client
        self._is_running = True
        self._fps = fps
        self._dtime = 0 if fps == 0 else 1 / fps

    def run(self):
        if self._client == ClientTypes.SIMULATION:
            self._run_simulation()
        else:
            self._run_vehicle()

    def _run_simulation(self):        
        CLIP = 7500
        while self._is_running:
            try:
                render_queue_data = self._data_queue.get(timeout=1)
                simulation_data:SimulationDataFrame = render_queue_data[0]
                control_data:ControlDataFrame = render_queue_data[1]
            except Empty:
                continue

            # image_rgb = cv2.convertScaleAbs(image_rgb, alpha=ALPHA, beta=BETA)
            qimage_rgb = QImage(simulation_data.camera_frame_rgb, SimulationCameraSettings.WIDTH, SimulationCameraSettings.HEIGHT, QImage.Format_RGB888)

            # byte1 = image_rgba[:, :, 3].astype(np.uint16)
            # byte2 = image_rgba[:, :, 4].astype(np.uint16)
            # combined = (byte2 << 8) | byte1
            # float_data = combined.view(np.float16)
            clipped_float_data = np.clip(simulation_data.camera_frame_depth, 0, CLIP)
            normalized_data = clipped_float_data / CLIP
            int8_data = (normalized_data * 255).astype(np.uint8)
            depth_map = cv2.applyColorMap(int8_data, cv2.COLORMAP_JET)
            qimage_depth = QImage(
                depth_map.data,
                SimulationCameraSettings.WIDTH,
                SimulationCameraSettings.HEIGHT,
                QImage.Format.Format_RGB888
            )

            # Emit
            self.data_ready.emit(
                (
                    simulation_data.speed,
                    control_data.speed_setpoint,
                    simulation_data.steering_angle,
                    control_data.steering_angle_setpoint,
                    simulation_data.pose.position.x,
                    simulation_data.pose.position.y,
                    simulation_data.pose.rotation.yaw,
                    qimage_rgb,
                    qimage_depth,
                )
            )

            # time.sleep(self._dtime)
    def _run_vehicle(self):
        CLIP = 7500
        while self._is_running:
            try:
                render_queue_data = self._data_queue.get(timeout=1)
                vehicle_data:VehicleDataFrame = render_queue_data[0]
                control_data:ControlDataFrame = render_queue_data[1]
            except Empty:
                continue

            qimage_rgb = QImage(vehicle_data.camera_frame_rgb, VehicleCamera.WIDTH, VehicleCamera.HEIGHT, QImage.Format_RGB888)
            clipped_float_data = np.clip(vehicle_data.camera_frame_depth, 0, CLIP)
            normalized_data = clipped_float_data / CLIP
            int8_data = (normalized_data * 255).astype(np.uint8)
            depth_map = cv2.applyColorMap(int8_data, cv2.COLORMAP_JET)
            qimage_depth = QImage(depth_map.data, VehicleCamera.WIDTH, VehicleCamera.HEIGHT, QImage.Format.Format_RGB888)

            # Emit
            self.data_ready.emit(
                (
                    vehicle_data,
                    control_data,
                    qimage_rgb,
                    qimage_depth,
                )
            )

    def stop(self):
        self._is_running = False


class SimulationRendererApp(QWidget):
    def __init__(self):
        super().__init__()
        # Window and App
        # ---------------------------------------
        self.setWindowTitle("RoboSim Data View")
        # self.setWindowFlags(Qt.FramelessWindowHint)

        self._drag_pos = None

        # Plotting
        # ---------------------------------------
        self.rgb_label = QLabel(self)
        # self.rgb_label.setMinimumSize(SimulationCameraSettings.WIDTH, SimulationCameraSettings.HEIGHT)
        self.rgb_pixmap = QPixmap()

        self.depth_label = QLabel(self)
        # self.depth_label.setMinimumSize(SimulationCameraSettings.WIDTH, SimulationCameraSettings.HEIGHT)
        self.depth_pixmap = QPixmap()

        self.speed_plot = pg.PlotWidget()
        self.speed_plot.setMinimumSize(SimulationCameraSettings.WIDTH, SimulationCameraSettings.HEIGHT)
        self.speed_plot.setXRange(-DATA_QUEUE_SIZE, 0)
        # self.speed_plot.setYRange(-1, 1)
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
            name="Value"
        )
        self.speed_setpoint_plot_data = self.speed_plot.plot(
            PLOT_TIME_STEPS,
            self.speed_setpoint_data,
            pen=pg.mkPen(QColor(0, 255, 0, 64), style=Qt.DashLine),
            name="Setpoint"
        )

        self.steering_plot = pg.PlotWidget()
        self.steering_plot.setMinimumSize(SimulationCameraSettings.WIDTH, SimulationCameraSettings.HEIGHT)
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
            name="Value"
        )
        self.steering_setpoint_plot_data = self.steering_plot.plot(
            self.steering_setpoint_data,
            PLOT_TIME_STEPS,
            pen=pg.mkPen(QColor(255, 0, 0, 128), style=Qt.DashLine),
            name="Setpoint"
        )
        
        self.map_plot = pg.PlotWidget()
        self.map_plot.setMinimumSize(SimulationCameraSettings.WIDTH, SimulationCameraSettings.HEIGHT)  # Adjust the size as needed
        self.map_plot.setXRange(-8000, 8000)  # Adjust the range as needed
        self.map_plot.setYRange(-8000, 8000)  # Adjust the range as needed
        self.map_plot.getPlotItem().showGrid(x=True, y=True)
        self.map_plot.getPlotItem().setTitle("Vehicle Position (X, Y, Yaw)")
        self.arrow = pg.ArrowItem(angle=90, tipAngle=30, baseAngle=20, headLen=20, tailLen=10, headWidth=10, tailWidth=4, pen={'color': 'g', 'width': 2})
        self.map_plot.addItem(self.arrow)
        self.arrow.setZValue(1)
        
        # Create a deque to store the past positions
        self.map_plot_positions_x = deque(maxlen=DATA_QUEUE_SIZE)
        self.map_plot_positions_y = deque(maxlen=DATA_QUEUE_SIZE)

        # Create a PlotCurveItem to represent the path
        self.path = pg.PlotCurveItem(pen=pg.mkPen(QColor(255, 255, 255, 255), width=2, style=Qt.DashLine))
        gradient = QLinearGradient(0, 0, 0, 1)
        gradient.setColorAt(0, QColor(255, 255, 255, 255))
        gradient.setColorAt(1, QColor(255, 255, 255, 64))
        self.path.setBrush(QBrush(gradient))
        
        self.map_plot.addItem(self.path)
        

        layout = QGridLayout(self)
        
        # layout.addWidget(self.map_plot, 0, 0)
        # layout.addWidget(self.rgb_label, 0, 1)
        # layout.addWidget(self.depth_label, 1, 1)
        # layout.addWidget(self.speed_plot, 0, 2)
        # layout.addWidget(self.steering_plot, 1, 2)
        
        layout.addWidget(self.rgb_label, 0, 0)
        layout.addWidget(self.depth_label, 1, 0)
        layout.addWidget(self.speed_plot, 0, 1)
        layout.addWidget(self.steering_plot, 1, 1)
        self.setStyleSheet("background-color: #2d2a2e;")
        

    def update_image(self, data):
        (
            speed,
            speed_setpoint,
            steering,
            steering_setpoint,
            x,
            y,
            yaw,
            qimage_rgb,
            qimage_depth,
        ) = data
        # RGB
        self.rgb_pixmap.convertFromImage(qimage_rgb)
        self.rgb_label.setPixmap(self.rgb_pixmap)
        # Depth
        self.depth_pixmap.convertFromImage(qimage_depth)
        self.depth_label.setPixmap(self.depth_pixmap)
        # Plot speed
        self.speed_data.append(speed)
        self.speed_plot_data.setData(PLOT_TIME_STEPS, self.speed_data)
        self.speed_marker.setData([PLOT_TIME_STEPS[-1]], [speed])
        self.speed_setpoint_data.append(speed_setpoint)
        self.speed_setpoint_plot_data.setData(PLOT_TIME_STEPS, self.speed_setpoint_data)
        # Plot steering
        self.steering_data.append(steering)
        self.steering_plot_data.setData(self.steering_data, PLOT_TIME_STEPS)
        self.steering_marker.setData([steering], [PLOT_TIME_STEPS[-1]])
        self.steering_setpoint_data.append(steering_setpoint)
        self.steering_setpoint_plot_data.setData(
            self.steering_setpoint_data, PLOT_TIME_STEPS
        )
        # Arrow
        self.arrow.setPos(y, x)
        self.arrow.setStyle(angle=yaw+90)
        # Add the new position to the deque
        self.map_plot_positions_x.append(x)
        self.map_plot_positions_y.append(y)

        self.path.setData(list(self.map_plot_positions_y), list(self.map_plot_positions_x))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = None

RAW2DEG = 360/4096

class VehicleRendererApp(QWidget):
    def __init__(self):
        super().__init__()
        # Window and App
        # ---------------------------------------
        self.setWindowTitle("ToySim Data View")
        # self.setWindowFlags(Qt.FramelessWindowHint)

        self._drag_pos = None

        # Plotting
        # ---------------------------------------
        self.rgb_label = QLabel(self)
        self.rgb_label.setMinimumSize(VehicleCamera.WIDTH, VehicleCamera.HEIGHT)
        self.rgb_pixmap = QPixmap()

        self.depth_label = QLabel(self)
        self.depth_label.setMinimumSize(VehicleCamera.WIDTH, VehicleCamera.HEIGHT)
        self.depth_pixmap = QPixmap()

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
            name="Value"
        )
        self.speed_setpoint_plot_data = self.speed_plot.plot(
            PLOT_TIME_STEPS,
            self.speed_setpoint_data,
            pen=pg.mkPen(QColor(0, 255, 0, 64), style=Qt.DashLine),
            name="Setpoint"
        )

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
            name="Value"
        )
        self.steering_setpoint_plot_data = self.steering_plot.plot(
            self.steering_setpoint_data,
            PLOT_TIME_STEPS,
            pen=pg.mkPen(QColor(255, 0, 0, 128), style=Qt.DashLine),
            name="Setpoint"
        )
        
        self.map_plot = pg.PlotWidget()
        self.map_plot.setXRange(-100, 100)  # Adjust the range as needed
        self.map_plot.setYRange(-100, 100)  # Adjust the range as needed
        self.map_plot.getPlotItem().showGrid(x=True, y=True)
        self.map_plot.getPlotItem().setTitle("Vehicle Position (X, Y, Yaw)")
        # self.arrow = pg.ArrowItem(angle=90, tipAngle=30, baseAngle=20, headLen=20, tailLen=10, headWidth=10, tailWidth=4, pen={'color': 'g', 'width': 2})
        # self.map_plot.addItem(self.arrow)
        # self.arrow.setZValue(1)
        self.orientation_line1 = pg.PlotCurveItem(pen=pg.mkPen('r', width=2))
        self.orientation_line1.setZValue(1)
        self.orientation_line2 = pg.PlotCurveItem(pen=pg.mkPen('g', width=2))
        self.orientation_line2.setZValue(1)
        self.map_plot.addItem(self.orientation_line1)
        self.map_plot.addItem(self.orientation_line2)
        
        # Create a deque to store the past positions
        self.map_plot_positions_x = deque(maxlen=DATA_QUEUE_SIZE)
        self.map_plot_positions_y = deque(maxlen=DATA_QUEUE_SIZE)

        # Create a PlotCurveItem to represent the path
        self.path = pg.PlotCurveItem(pen=pg.mkPen(QColor(255, 255, 255, 255), width=2, style=Qt.DashLine))
        gradient = QLinearGradient(0, 0, 0, 1)
        gradient.setColorAt(0, QColor(255, 255, 255, 255))
        gradient.setColorAt(1, QColor(255, 255, 255, 64))
        self.path.setBrush(QBrush(gradient))
        self.map_plot.addItem(self.path)
        
        
        self.imu_plot = gl.GLViewWidget()  # Placeholder for the 3D plot
        self.imu_plot.setSizePolicy(self.map_plot.sizePolicy())       
        self.init_imu_plot()
        self.left_encoder_plot = pg.PlotWidget()  # Placeholder for the first smaller plot
        self.init_left_encoder_plot()
        self.right_encoder_plot = pg.PlotWidget()  # Placeholder for the second smaller plot

        # Create the main grid layout
        main_grid_layout = QGridLayout()
        main_grid_layout.addWidget(self.rgb_label)
        main_grid_layout.addWidget(self.depth_label)
        
        imu_layout = QHBoxLayout()
        imu_layout.addWidget(self.imu_plot)
        imu_layout.addWidget(self.map_plot)
        
        encoders_layout = QHBoxLayout()
        encoders_layout.addWidget(self.left_encoder_plot)
        encoders_layout.addWidget(self.right_encoder_plot)
        
        # Create a vertical layout for the left side
        left_layout = QVBoxLayout()
        left_layout.addLayout(imu_layout)
        left_layout.addLayout(encoders_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.speed_plot)
        right_layout.addWidget(self.steering_plot)

        # Create the main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(main_grid_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

        self.setStyleSheet("background-color: #2d2a2e;")
        

    def init_imu_plot(self):
        self.imu_plot.setCameraPosition(pos=QVector3D(0, 0, 0), distance=10, azimuth=225)
        self.imu_plot.setBackgroundColor('k')

        # Create the arrows using GLLinePlotItem
        self.arrow_x = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [1, 0, 0]]), color=(1, 0, 0, 1), width=3)
        self.arrow_y = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 1, 0]]), color=(0, 1, 0, 1), width=3)
        self.arrow_z = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 1]]), color=(0, 0, 1, 1), width=3)

        # Add the arrows to the plot
        self.imu_plot.addItem(self.arrow_x)
        self.imu_plot.addItem(self.arrow_y)
        self.imu_plot.addItem(self.arrow_z)
            
        grid = gl.GLGridItem()
        grid.setSize(x=10, y=10, z=10)
        self.imu_plot.addItem(grid)

        # Add axis labels using GLLinePlotItem
        x_label = gl.GLLinePlotItem(pos=np.array([[1, 0, 0], [1.1, 0, 0]]), color=(1, 0, 0, 1), width=3)
        y_label = gl.GLLinePlotItem(pos=np.array([[0, 1, 0], [0, 1.1, 0]]), color=(0, 1, 0, 1), width=3)
        z_label = gl.GLLinePlotItem(pos=np.array([[0, 0, 1], [0, 0, 1.1]]), color=(0, 0, 1, 1), width=3)
        self.imu_plot.addItem(x_label)
        self.imu_plot.addItem(y_label)
        self.imu_plot.addItem(z_label)

    def init_left_encoder_plot(self):
        self.left_encoder_plot.setAspectLocked()
        self.left_encoder_plot.setXRange(-500, 500)
        self.left_encoder_plot.setYRange(-500, 500)
        self.left_encoder_plot.getPlotItem().showGrid(x=True, y=True)
        self.left_encoder_plot.getPlotItem().setTitle("Left Encoder Data")

        # Create a deque to store the encoder data
        self.left_encoder_data = deque(maxlen=150)

        # Create a scatter plot item for the data points
        self.left_encoder_scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.left_encoder_plot.addItem(self.left_encoder_scatter)

        # Create a plot curve item to connect the dots
        self.left_encoder_curve = pg.PlotCurveItem(pen=pg.mkPen('w', width=1))
        self.left_encoder_plot.addItem(self.left_encoder_curve)

    def update_orientation_lines(self, x, y, yaw):
        length = 20  # Length of the orientation lines
        dx = length * np.cos(np.radians(yaw))
        dy = length * np.sin(np.radians(yaw))

        # Line 1: From (x, y) to (x + dx, y + dy)
        self.orientation_line1.setData([x, x + dx], [y, y + dy])

        # Line 2: Perpendicular to Line 1
        self.orientation_line2.setData([x, x - dy], [y, y + dx])

    def update_image(self, data):
        (
            vehicle_data,
            control_data,
            qimage_rgb,
            qimage_depth,
        ) = data
        # RGB
        self.rgb_pixmap.convertFromImage(qimage_rgb)
        self.rgb_label.setPixmap(self.rgb_pixmap)
        # Depth
        self.depth_pixmap.convertFromImage(qimage_depth)
        self.depth_label.setPixmap(self.depth_pixmap)
        # Plot speed
        speed_cm = vehicle_data.speed / 10
        self.speed_data.append(speed_cm)
        self.speed_plot_data.setData(PLOT_TIME_STEPS, self.speed_data)
        self.speed_marker.setData([PLOT_TIME_STEPS[-1]], [speed_cm])
        self.speed_setpoint_data.append(control_data.speed_setpoint)
        self.speed_setpoint_plot_data.setData(PLOT_TIME_STEPS, self.speed_setpoint_data)
        # Plot steering
        self.steering_data.append(vehicle_data.steering_angle)
        self.steering_plot_data.setData(self.steering_data, PLOT_TIME_STEPS)
        self.steering_marker.setData([vehicle_data.steering_angle], [PLOT_TIME_STEPS[-1]])
        self.steering_setpoint_data.append(control_data.steering_angle_setpoint)
        self.steering_setpoint_plot_data.setData(self.steering_setpoint_data,PLOT_TIME_STEPS)
        # Arrow
        # self.arrow.setPos(vehicle_data.pose.position.y, vehicle_data.pose.position.x)
        # self.arrow.setStyle(angle=vehicle_data.pose.rotation.yaw+90)
        # Add the new position to the deque
        self.map_plot_positions_x.append(vehicle_data.pose.position.x)
        self.map_plot_positions_y.append(vehicle_data.pose.position.y)

        self.path.setData(list(self.map_plot_positions_x), list(self.map_plot_positions_y))

        # Update IMU plot
        self.update_imu_plot(vehicle_data.pose.rotation)
        self.update_left_encoder_plot(vehicle_data.encoder_data)
        self.update_orientation_lines(vehicle_data.pose.position.x, vehicle_data.pose.position.y, vehicle_data.pose.rotation.yaw)

    def update_imu_plot(self, rotation):
        roll, pitch, yaw = np.deg2rad(rotation.roll), np.deg2rad(rotation.pitch), np.deg2rad(rotation.yaw)
        # Create rotation matrices
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(roll), -np.sin(roll), 0],
            [0, np.sin(roll), np.cos(roll), 0],
            [0, 0, 0, 1]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch), 0],
            [0, 1, 0, 0],
            [-np.sin(pitch), 0, np.cos(pitch), 0],
            [0, 0, 0, 1]
        ])

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0, 0],
            [np.sin(yaw), np.cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Update arrow orientations
        self.arrow_x.setTransform(R)
        self.arrow_y.setTransform(R)
        self.arrow_z.setTransform(R)

    def update_left_encoder_plot(self, encoder_data):
        # Convert polar coordinates to Cartesian coordinates
        position, magnitude = encoder_data.position, encoder_data.magnitude
        rad = np.deg2rad(position * RAW2DEG)
        x = magnitude * np.cos(rad)
        y = magnitude * np.sin(rad)

        # Append the new data point to the deque
        self.left_encoder_data.append((x, y))

        # Update the scatter plot and curve with the new data
        self.left_encoder_scatter.setData([p[0] for p in self.left_encoder_data], [p[1] for p in self.left_encoder_data])
        self.left_encoder_curve.setData([p[0] for p in self.left_encoder_data], [p[1] for p in self.left_encoder_data])

class Renderer:
    def __init__(self, data_queue: Queue, client):
        self._data_queue = data_queue
        self._client = client
        self._app = QApplication(sys.argv)
        self._data_thread = RendererDataThread(data_queue, client=client)
        self._app.aboutToQuit.connect(self._stop_thread_and_wait)

    def run(self):
        ex: QWidget = None
        if self._client == ClientTypes.SIMULATION:
            ex = SimulationRendererApp()
        else:
            ex = VehicleRendererApp()
        self._data_thread.finished.connect(self._app.exit)
        self._data_thread.data_ready.connect(ex.update_image)
        self._data_thread.start()
        ex.show()
        return self._app.exec()

    def _stop_thread_and_wait(self):
        self._data_thread.stop()
        self._data_thread.wait()
