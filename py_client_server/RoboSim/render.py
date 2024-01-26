import cv2
import numpy as np

import threading
from queue import Queue, Empty

from .settings import CAMERA_X, CAMERA_Y, CAMERA_PIXEL_COMPONENTS

# Contrast control (1.0-3.0)
ALPHA   = 3   
# Brightness control (0-100)
BETA    = 10

class RendererThread:
    def __init__(self, data_queue:Queue, exit_event:threading.Event, verbose=True):
        self._verbose = verbose
        self._data_queue = data_queue
        self._exit_event = exit_event
        self._thread = threading.Thread(target=self._loop, daemon=True)
    
    def start(self):
        self._thread.start()
        
    def join(self):
        self._thread.join()    
    
    def _loop(self):
        cv2.namedWindow("Feed Video", cv2.WINDOW_NORMAL)
        while True:
            if self._exit_event.is_set():
                break
            try:
                pixel_data:bytes = self._data_queue.get(timeout=1)
                self._render(pixel_data)        
            except Empty:
                pass
        cv2.destroyAllWindows()
    
    def _render(self, pixel_data:bytes):
        pixel_data_np = np.frombuffer(pixel_data, dtype=np.uint8)
        image_rgb = pixel_data_np.reshape((CAMERA_Y, CAMERA_X, CAMERA_PIXEL_COMPONENTS))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)
        image_mastered = cv2.convertScaleAbs(image_bgr, alpha=ALPHA, beta=BETA)
        cv2.imshow("Feed Video", image_mastered)
        cv2.waitKey(1)
        


import cv2
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

class AppThread(QThread):
    def __init__(self, parent, data_queue:Queue, exit_event:threading.Event):
        QThread.__init__(self, parent=parent)
        self._data_queue = data_queue
        self._exit_event = exit_event
        
    changePixmap = pyqtSignal(QImage)
    def run(self):
        while not self._exit_event.is_set():  
            try:
                pixel_data:bytes = self._data_queue.get(timeout=1)
                # print(len(pixel_data))
                pixel_data_np = np.frombuffer(pixel_data, dtype=np.uint8)
                image_rgb = pixel_data_np.reshape((CAMERA_Y, CAMERA_X, CAMERA_PIXEL_COMPONENTS))
                rgbImage = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)      
            except Empty:
                pass


class App(QWidget):
    def __init__(self, data_queue:Queue, exit_event:threading.Event):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.w = 640
        self.h = 480
        self._data_queue = data_queue
        self._exit_event = exit_event
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.w, self.h)
        self.resize(1800, 1200)
        # create a label
        self.label = QLabel(self)
        self.label.move(280, 120)
        self.label.resize(640, 480)
        th = AppThread(self, self._data_queue, self._exit_event)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()