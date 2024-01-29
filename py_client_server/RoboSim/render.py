import sys
import time
import cv2
import numpy as np
from collections import deque
from queue import Queue, Empty

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QWidget
import pyqtgraph as pg

from .settings import CAMERA_X, CAMERA_Y


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
ALPHA   = 3   
# Brightness control (0-100)
BETA    = 10


class RendererDataThread(QThread):
    data_ready = Signal(tuple)

    def __init__(self, data_queue: Queue, fps=FPS):
        super().__init__()
        self._data_queue = data_queue
        self._is_running = True
        self._fps = fps
        self._dtime = 0 if fps == 0 else 1 / fps

    def run(self):
        while self._is_running:
            try:
                (
                    speed,
                    speed_setpoint,
                    steering_angle,
                    steering_angle_setpoint,
                    image_rgba,
                ) = self._data_queue.get(timeout=1)
            except Empty:
                continue

            image_rgb = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2RGB)
            # TODO: move to settings
            image_mastered = cv2.convertScaleAbs(image_rgb, alpha=ALPHA, beta=BETA)
            qimage_rgb = QImage(image_mastered, CAMERA_X, CAMERA_Y, QImage.Format_RGB888)

            depth_map = image_rgba[:, :, 3]
            depth_map = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_JET
            )
            qimage_depth = QImage(
                depth_map.data, CAMERA_X, CAMERA_Y, QImage.Format_RGB888
            )

            # Emit
            self.data_ready.emit((speed, steering_angle, qimage_rgb, qimage_depth))
            time.sleep(self._dtime)

    def stop(self):
        self._is_running = False
        

class RendererApp(QWidget):
    def __init__(self):
        super().__init__()
        # Window and App
        # ---------------------------------------
        self.setWindowTitle("RoboSim Data View")
        self.setWindowFlags(Qt.FramelessWindowHint)

        self._drag_pos = None

        # Plotting
        # ---------------------------------------
        self.rgb_label = QLabel(self)
        self.rgb_pixmap = QPixmap()

        self.depth_label = QLabel(self)
        self.depth_pixmap = QPixmap()

        self.speed_plot = pg.PlotWidget()
        self.speed_plot.setMinimumSize(640, 480)
        self.speed_plot.setXRange(-DATA_QUEUE_SIZE, 0)
        self.speed_plot.setYRange(-1, 1)
        self.speed_plot.getAxis("bottom").setTicks(STEP_TICKS)
        self.speed_plot.getPlotItem().showGrid(x=True, y=True)
        self.speed_plot.getPlotItem().setTitle("Speed")
        self.speed_plot.getPlotItem().setLabel("left", "Speed [actual/max]")
        self.speed_plot.getPlotItem().setLabel("bottom", f"Step [n] (s = {FPS} steps)")
        self.speed_marker = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush="w")
        self.speed_marker.setZValue(1)
        self.speed_plot.addItem(self.speed_marker)
        self.speed_data = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)
        self.speed_plot_data = self.speed_plot.plot(PLOT_TIME_STEPS, self.speed_data, pen="g")

        self.steering_plot = pg.PlotWidget()
        self.steering_plot.setMinimumSize(640, 480)
        self.steering_plot.setXRange(-40, 40)
        self.steering_plot.getAxis("left").setTicks(STEP_TICKS)
        self.steering_plot.getPlotItem().showGrid(x=True, y=True)
        self.steering_plot.getPlotItem().setTitle("Steering Angle")
        self.steering_plot.getPlotItem().setLabel("left", f"Step [n] (s = {FPS} steps)")
        self.steering_plot.getPlotItem().setLabel("bottom", "Steering angle [deg]")
        self.steering_data = deque(PLOT_QUEUE_DEFAULT_DATA, maxlen=DATA_QUEUE_SIZE)
        self.steering_marker = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush="w")
        self.steering_marker.setZValue(1)
        self.steering_plot.addItem(self.steering_marker)
        self.steering_plot_data = self.steering_plot.plot(self.steering_data, PLOT_TIME_STEPS, pen="r" )

        layout = QGridLayout(self)
        layout.addWidget(self.rgb_label, 0, 0)
        layout.addWidget(self.depth_label, 1, 0)
        layout.addWidget(self.speed_plot, 0, 1)
        layout.addWidget(self.steering_plot, 1, 1)
        self.setStyleSheet("background-color: #222222;")

    def update_image(self, data):
        speed, steering_angle, qimage_rgb, qimage_depth = data
        # RGB
        self.rgb_pixmap.convertFromImage(qimage_rgb)
        self.rgb_label.setPixmap(self.rgb_pixmap)
        # Depth
        self.depth_pixmap.convertFromImage(qimage_depth)
        self.depth_label.setPixmap(self.depth_pixmap)
        # Plot
        self.speed_data.append(speed)
        self.steering_data.append(steering_angle)
        self.speed_plot_data.setData(PLOT_TIME_STEPS, self.speed_data)
        self.steering_plot_data.setData(self.steering_data, PLOT_TIME_STEPS)
        self.speed_marker.setData([PLOT_TIME_STEPS[-1]], [speed])
        self.steering_marker.setData([steering_angle], [PLOT_TIME_STEPS[-1]])

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


class Renderer():
    def __init__(self, data_queue:Queue):
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
