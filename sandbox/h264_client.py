import numpy as np
import socket
import time

class Data:
    def __init__(self):
        self._counter = 0

    def __call__(self):
        data = np.zeros((720, 1280, 3), dtype=np.uint8)
        data[self._counter : self._counter + 10, :, 1] = 255
        self._counter = (self._counter + 5) % 720 
        return data.tobytes()
if __name__ == '__main__':
    addr = ('localhost', 8888)
    data = Data()
    dt = 1/30
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(addr)
        while True:
            time.sleep(dt)
            sock.sendall(data())