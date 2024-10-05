import sys
import time
import cv2
import numpy as np
from collections import deque
from queue import Queue, Empty

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QLinearGradient
from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QWidget
import pyqtgraph as pg

from .settings import SimulationCameraSettings
from .processor import SimulationDataFrame, ControlDataFrame

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

    def __init__(self, data_queue: Queue, fps=FPS):
        super().__init__()
        self._data_queue = data_queue
        self._is_running = True
        self._fps = fps
        self._dtime = 0 if fps == 0 else 1 / fps

    def run(self):
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

    def stop(self):
        self._is_running = False


class RendererApp(QWidget):
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


class Renderer:
    def __init__(self, data_queue: Queue):
        self._data_queue = data_queue
        self._app = QApplication(sys.argv)
        self._network_thread = RendererDataThread(data_queue)
        self._app.aboutToQuit.connect(self._stop_thread_and_wait)

    def run(self):
        ex = RendererApp()
        self._network_thread.finished.connect(self._app.exit)
        self._network_thread.data_ready.connect(ex.update_image)
        self._network_thread.start()
        ex.show()
        return self._app.exec()

    def _stop_thread_and_wait(self):
        self._network_thread.stop()
        self._network_thread.wait()
