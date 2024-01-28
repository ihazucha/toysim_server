import sys
import time

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QWidget
from RoboSim.settings import CAMERA_PIXEL_COMPONENTS, CAMERA_X, CAMERA_Y

from collections import deque

FPS = 30
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
            data = np.zeros((CAMERA_Y, CAMERA_X, CAMERA_PIXEL_COMPONENTS), dtype=np.uint8)
            data[self._counter:self._counter+10, :, 1] = 255
            self._counter = (self._counter + 1) % CAMERA_Y
            
            speed = np.sin(self.time_step)
            steering_angle = 40 * np.sin(self.time_step)
            self.data_ready.emit((speed, steering_angle, data))
            self.time_step += DTIME
            time.sleep(DTIME) 
    
    def stop(self):
        self._is_running = False


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.rgb_label = QLabel(self)
        self.depth_label = QLabel(self)
        
        plot_queue_data = list([0 for _ in range(DATA_QUEUE_SIZE)])
        self._plot_time_steps = np.arange(-DATA_QUEUE_SIZE, 0, 1)
        
        self.speed_plot = pg.PlotWidget()
        self.speed_plot.setMinimumSize(640, 480)
        self.speed_plot.setYRange(-1, 1)
        self.speed_plot.getPlotItem().showAxis('right')
        self.speed_plot.getPlotItem().hideAxis('left')
        self.speed_plot.getPlotItem().showGrid(x=True, y=True)

        self.speed_data = deque(plot_queue_data, maxlen=DATA_QUEUE_SIZE)
        
        self.steering_plot = pg.PlotWidget()
        self.steering_plot.setMinimumSize(640, 480)
        self.steering_plot.setXRange(-40, 40)
        self.steering_plot.getPlotItem().showAxis('top')
        self.steering_plot.getPlotItem().hideAxis('bottom')
        self.steering_plot.getPlotItem().showGrid(x=True, y=True)
        
        self.steering_data = deque(plot_queue_data, maxlen=DATA_QUEUE_SIZE)

        self.speed_marker = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255))
        self.steering_marker = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255))
        

        layout = QGridLayout(self)
        layout.addWidget(self.rgb_label, 0, 0)
        layout.addWidget(self.depth_label,1, 0)
        layout.addWidget(self.speed_plot, 1, 1)
        layout.addWidget(self.steering_plot, 0, 1)
        
        self.setStyleSheet("background-color: #222222;")

    def update_image(self, data):
        speed, steering_angle, data = data
        
        # RGB
        image_rgb = cv2.cvtColor(data, cv2.COLOR_RGBA2RGB)
        image = QImage(image_rgb, CAMERA_X, CAMERA_Y, QImage.Format_RGB888)
        self.rgb_label.setPixmap(QPixmap.fromImage(image))
        # Depth
        depth_map = data[:, :, 3]  # Assuming the A component is the depth map
        depth_map = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_JET)
        qimage_depth = QImage(depth_map.data, CAMERA_X, CAMERA_Y, QImage.Format_RGB888)
        self.depth_label.setPixmap(QPixmap.fromImage(qimage_depth))
        # Plot          
        self.speed_data.append(speed)
        self.steering_data.append(steering_angle)
        self.speed_plot.plot(self._plot_time_steps, self.speed_data, clear=True, pen='g')
        self.steering_plot.plot(self.steering_data, self._plot_time_steps, clear=True, pen='r')
        # Add a ScatterPlotItem at the last point of the speed plot
        self.speed_marker.setData([self._plot_time_steps[-1]], [speed])
        self.speed_plot.addItem(self.speed_marker)
        self.steering_marker.setData([steering_angle], [self._plot_time_steps[-1]])
        self.steering_plot.addItem(self.steering_marker)


    
def main():
    app = QApplication(sys.argv)
    ex = App()
    network_thread = NetworkDataThread()
    network_thread.data_ready.connect(ex.update_image)
    network_thread.start()
    app.aboutToQuit.connect(network_thread.stop)
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()