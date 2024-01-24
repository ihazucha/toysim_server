#!/usr/bin/env python3

import socket
import struct
import numpy as np
import cv2
import time
import collections

import threading
import queue


HOST = 'localhost'
PORT = 3333
ADDR = (HOST, PORT)

CLIENT_ADDR = (HOST, 6666)

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

         
class CarSimUdpServer:
    DATA_HEADER_SIZE    = 4
    CAMERA_DATA_SIZE    = 3145728
    
    def __init__(self, verbose=True):
        self._verbose = verbose
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self._data_queue = queue.Queue(maxsize=2)
        self._receive_thread_started = threading.Event()
        self._render_thread_started = threading.Event()
        self._receive_thread = threading.Thread(target=self.receive_loop, daemon=True)
        self._receive_thread.start()
        self._render_thread = threading.Thread(target=self.render_loop, daemon=True)
        self._render_thread.start()
                
    def __del__(self):
        self._close()

    def _close(self):
        if self._sock is None:
            return
        try:
            self._log(f"Closing socket...")
            self._sock.shutdown(socket.SHUT_RDWR)
            self._sock.close()
            self._log(f"Socket closed")
        except OSError as e:
            print(e)
            pass
    
    def _log(self, msg:str, who:str="[Conn]"):
        if self._verbose: print(f"{who} {msg}")
    
    def _send_settings(self):
        self._log(f"Sending settings...")
        self._sock.sendto(b"settings", CLIENT_ADDR)
    
    def _parse_data_header(self, header: bytes):
        payload_size, = struct.unpack("i", header)
        return payload_size
        
    def _parse_data_payload(self, payload: bytes):
        # 8-bit 1024x1024 image
        image = np.frombuffer(payload, dtype=np.uint8)
        image = image.reshape((1024, 1024, 3))
        return image
    
    def _recv_data(self):
        header = self._sock.recv(self.__class__.DATA_HEADER_SIZE)
        payload_size = self._parse_data_header(header)
        print(payload_size)
        if payload_size == 0:
            return None
        payload = self._sock.recv(payload_size)
        payload_parsed = self._parse_data_payload(payload)
        return payload_parsed

    def receive_loop(self):
        self._receive_thread_started.set()
        self._render_thread_started.wait()
        self._send_settings()
        while True:
            if not self._render_thread.is_alive():
                break
            try:
                print("Hop")
                data = self._recv_data()
                self._data_queue.put(data)
            except OSError as e:
                print(e)
                self._log("Connection closed by client")
                break
    
    def render_loop(self):
        self._render_thread_started.set()
        self._receive_thread_started.wait()
        cv2.namedWindow("Feed Video", cv2.WINDOW_NORMAL)
        while True:
            if not self._receive_thread.is_alive():
                cv2.destroyAllWindows()
                break
            try:
                data = self._data_queue.get(timeout=1)
                cv2.imshow("Feed Video", data)
                cv2.waitKey(1)
            except queue.Empty:
                cv2.destroyAllWindows()
            
    def wait(self):
        self._log(f"Waiting for recv thread to finish...")
        self._receive_thread.join()
        self._log(f"Waiting for render thread to finish...")
        self._render_thread.join()
        self._log(f"Threads finished")


if __name__ == "__main__":

    try:
        server = CarSimUdpServer()
        server.wait()
    except KeyboardInterrupt:
        print("[Main] Exiting...")
        pass
    finally:
        print("[Main] Cleanup...")
        cv2.destroyAllWindows()