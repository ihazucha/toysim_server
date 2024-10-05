import collections
import time


class PerformanceMonitor:
    WINDOW_LENGTH = 10
    
    def __init__(self, name:str, avg_sample_count:int=WINDOW_LENGTH):
        self._name = name
        self._avg_sample_count = avg_sample_count
        self._frame_times = collections.deque(maxlen=avg_sample_count)
        self._counter:int = 0
    
    def __call__(self, frame_time:int|float):
        self._frame_times.append(frame_time)
        self._counter += 1
        if self._counter == self.__class__.WINDOW_LENGTH:
            avg_fps = self._avg_sample_count / sum(self._frame_times)
            print(f"{self._name}: {avg_fps}")
            self._counter = 0
            
    def start(self):
        self._start_time = time.time()
        
    def end(self):
        self(time.time() - self._start_time)