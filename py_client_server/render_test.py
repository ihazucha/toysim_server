import cv2
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

from RoboSim.settings import CAMERA_X, CAMERA_Y, CAMERA_PIXEL_COMPONENTS
import threading
from queue import Queue, Empty
import sys
import numpy as np

class AppThread(QThread):
    changePixmap = pyqtSignal(QImage)
    def run(self):
        data = DummyImageData()
        while True:  
            try:
                pixel_data_np = data()
                image_rgb = pixel_data_np.reshape((CAMERA_Y, CAMERA_X, CAMERA_PIXEL_COMPONENTS))
                rgbImage = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)      
            except Empty:
                pass

class DummyImageData:
    def __init__(self):
        self._counter = 0
    def __call__(self):
        data = np.zeros((CAMERA_Y, CAMERA_X, CAMERA_PIXEL_COMPONENTS), dtype=np.uint8)
        data[self._counter:self._counter+10, :, 1] = 255
        self._counter = (self._counter + 5) % CAMERA_Y
        return data.flatten()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.w = 640
        self.h = 480
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
        th = AppThread()
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())