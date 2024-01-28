import sys
import time

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QWidget
from RoboSim.settings import CAMERA_PIXEL_COMPONENTS, CAMERA_X, CAMERA_Y

from collections import deque

FPS = 60
DTIME = 1 / FPS
DATA_QUEUE_LENGTH_SECONDS = 5
DATA_QUEUE_SIZE = FPS * DATA_QUEUE_LENGTH_SECONDS

class NetworkDataThread(QThread):
    data_ready = Signal(tuple)
    
    def __init__(self):
        super().__init__()
        self._counter = 0
        self._is_running = True
        self.time_step = 0

    def run(self):
        while self._is_running:
            # Images
            data = np.zeros((CAMERA_Y, CAMERA_X, CAMERA_PIXEL_COMPONENTS), dtype=np.uint8)
            data[self._counter:self._counter+10, :, 1] = 255
            self._counter = (self._counter + 1) % CAMERA_Y
            
            image_rgb = cv2.cvtColor(data, cv2.COLOR_RGBA2RGB)
            qimage_rgb = QImage(image_rgb, CAMERA_X, CAMERA_Y, QImage.Format_RGB888)
            
            depth_map = data[:, :, 3]  # Assuming the A component is the depth map
            depth_map = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_JET)
            qimage_depth = QImage(depth_map.data, CAMERA_X, CAMERA_Y, QImage.Format_RGB888)
            
            # Telemetry
            speed = np.sin(self.time_step)
            steering_angle = 40 * np.sin(self.time_step)
            
            # Emit
            self.data_ready.emit((speed, steering_angle, qimage_rgb, qimage_depth))
            
            self.time_step += DTIME
            time.sleep(DTIME) 
    
    def stop(self):
        self._is_running = False


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoboSim Data View")
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        self.rgb_label = QLabel(self)
        self.rgb_pixmap = QPixmap()
        
        self.depth_label = QLabel(self)
        self.depth_pixmap = QPixmap()
        
        plot_queue_data = list([0 for _ in range(DATA_QUEUE_SIZE)])
        self._plot_time_steps = np.arange(-DATA_QUEUE_SIZE, 0, 1)
        time_major_ticks = list(zip(range(-DATA_QUEUE_SIZE, 1, FPS), map(str, range(-DATA_QUEUE_SIZE, 1, FPS))))
        time_minor_ticks = list(zip(range(-DATA_QUEUE_SIZE, 1, FPS//2), ['' for _ in range(DATA_QUEUE_SIZE//2)]))
        
        self.speed_plot = pg.PlotWidget()
        self.speed_plot.setMinimumSize(640, 480)
        self.speed_plot.setXRange(-DATA_QUEUE_SIZE, 0)
        self.speed_plot.setYRange(-1, 1)
        self.speed_plot.getAxis('bottom').setTicks([time_major_ticks, time_minor_ticks])
        self.speed_plot.getPlotItem().showGrid(x=True, y=True)
        self.speed_plot.getPlotItem().setTitle('Speed')
        self.speed_plot.getPlotItem().setLabel('left', 'Speed [m/s]')
        self.speed_plot.getPlotItem().setLabel('bottom', f"Simulation step [n] (s = {FPS} steps)")
        self.speed_data = deque(plot_queue_data, maxlen=DATA_QUEUE_SIZE)
        self.speed_marker = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255))
        self.speed_marker.setZValue(1)
        self.speed_plot.addItem(self.speed_marker)
        self.speed_plot_data = self.speed_plot.plot(self._plot_time_steps, self.speed_data, pen='g')
          
        self.steering_plot = pg.PlotWidget()
        self.steering_plot.setMinimumSize(640, 480)
        self.steering_plot.setXRange(-40, 40)
        self.steering_plot.getAxis('left').setTicks([time_major_ticks, time_minor_ticks])
        self.steering_plot.getPlotItem().showGrid(x=True, y=True)
        self.steering_plot.getPlotItem().setTitle('Steering Angle')
        self.steering_plot.getPlotItem().setLabel('left', f"Simulation step [n] (s = {FPS} steps)")
        self.steering_plot.getPlotItem().setLabel('bottom', 'Steering angle [deg]')
        self.steering_data = deque(plot_queue_data, maxlen=DATA_QUEUE_SIZE)
        self.steering_marker = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255))
        self.steering_marker.setZValue(1)
        self.steering_plot.addItem(self.steering_marker)
        self.steering_plot_data = self.steering_plot.plot(self.steering_data, self._plot_time_steps, pen='r')

        layout = QGridLayout(self)
        layout.addWidget(self.rgb_label, 0, 0)
        layout.addWidget(self.depth_label, 1, 0)
        layout.addWidget(self.speed_plot, 0, 1)
        layout.addWidget(self.steering_plot, 1, 1)
        self.setStyleSheet("background-color: #222222;")
        
        self._drag_pos = None  # Store the position where the mouse click occurred

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = None

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
        self.speed_plot_data.setData(self._plot_time_steps, self.speed_data)
        self.steering_plot_data.setData(self.steering_data, self._plot_time_steps) 
        self.speed_marker.setData([self._plot_time_steps[-1]], [speed])
        self.steering_marker.setData([steering_angle], [self._plot_time_steps[-1]])


    
def main():
    app = QApplication(sys.argv)
    ex = App()
    
    network_thread = NetworkDataThread()
    network_thread.finished.connect(app.exit)
    network_thread.data_ready.connect(ex.update_image)
    network_thread.start()
    
    def stop_thread_and_wait():
        network_thread.stop()
        network_thread.wait()

    app.aboutToQuit.connect(stop_thread_and_wait)
    
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()