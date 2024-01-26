import cv2
import numpy as np

from threading import Thread, Event
from queue import Queue, Empty


class ImageDataRenderer:
    def __init__(self, data_queue:Queue, exit_event:Event):
        self._data_queue = data_queue
        self._render_thread = Thread(target=self._loop, daemon=True)
        self._exit_event = exit_event
    
    def start(self):
        self._render_thread.start()    
    
    def _loop(self):
        try:
            cv2.namedWindow("Feed Video", cv2.WINDOW_NORMAL)
            while True:
                if self._exit_event.is_set():
                    break
                data:bytes = self._data_queue.get(timeout=3)
                image = np.frombuffer(data, dtype=np.uint8)
                image = image.reshape((1024, 1024, 3))
                cv2.imshow("Feed Video", image)
                cv2.waitKey(1)
        except Empty:
            print("Empty")
            pass
        finally:
            self._exit_event.set()
            cv2.destroyAllWindows()