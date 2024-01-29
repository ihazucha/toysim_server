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