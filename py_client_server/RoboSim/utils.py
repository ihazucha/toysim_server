import collections
import time


class PerformanceMonitor:
    WINDOW_LENGTH = 60
    
    def __init__(self, avg_window_len=WINDOW_LENGTH):
        self._frame_times = collections.deque(maxlen=avg_window_len)
        self._counter:int = 0
    
    def __call__(self, frame_time):
        self._frame_times.append(frame_time)
        self._counter += 1
        if self._counter == self.__class__.WINDOW_LENGTH:
            avg_fps = len(self._frame_times) / sum(self._frame_times)
            print(f"FPS: {avg_fps}")
            self._counter = 0
            
    def start(self):
        self._start_time = time.time()
        
    def end(self):
        self(time.time() - self._start_time)