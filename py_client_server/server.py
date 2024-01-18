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
PORT = 6666

class CarSimListener:
    def __init__(self, host:str=HOST, port:int=PORT, verbose:bool=True):
        self._verbose = verbose
        self._listener_socket = None
        self._conn = None
        self._bind_listen((host, port))
        
    def _log(self, msg:str, who:str="[Listener]"):
        if self._verbose: print(f"{who} {msg}")

    def _bind_listen(self, addr:tuple):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(addr)
        self._sock.listen()
        self._log(f"Active on: {addr}")
   
    def await_connection(self):
        while True:
            self._log(f"Waiting for client...")
            sock, addr = self._sock.accept()
            self._conn = CarSimConnection(sock)
            self._log(f"Opening connection with {addr}")
            self._conn.wait()
            self._log(f"Closing connection with {addr}")
            
        
class CarSimConnection:
    DATA_HEADER_SIZE    = 4
    CAMERA_DATA_SIZE    = 3145728
    
    def __init__(self, sock, verbose=True):
        self._verbose = verbose
        self._sock = sock
        
        self._data_queue = queue.Queue(maxsize=2)
        self._receive_thread = threading.Thread(
            target=self.receive_loop, daemon=True)
        self._receive_thread.start()
        self._render_thread = threading.Thread(
            target=self.render_loop, daemon=True)
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
        except OSError:
            pass
    
    def _log(self, msg:str, who:str="[Conn]"):
        if self._verbose: print(f"{who} {msg}")
    
    def _send_settings(self):
        self._log(f"Sending settings...")
        self._sock.sendall(b"settings")

    def _recv_all(self, size):
        data = b''
        while len(data) < size:
            more = self._sock.recv(size - len(data))
            if not more:
                raise IOError('Socket closed before all data received')
            data += more
        return data
    
    def _parse_data_header(self, header: bytes):
        payload_size, = struct.unpack("i", header)
        return payload_size
        
    def _parse_data_payload(self, payload: bytes):
        # 8-bit 1024x1024 image
        image = np.frombuffer(payload, dtype=np.uint8)
        image = image.reshape((1024, 1024, 3))
        return image
    
    def _recv_data(self):
        header = self._recv_all(self.__class__.DATA_HEADER_SIZE)
        payload_size = self._parse_data_header(header)
        if payload_size == 0:
            return None
        payload = self._recv_all(payload_size)
        payload_parsed = self._parse_data_payload(payload)
        return payload_parsed

    def receive_loop(self):
        self._send_settings()
        MAXLEN = 60
        frame_times = collections.deque(maxlen=MAXLEN)
        counter = 0
        time.sleep(0.1) # To prevent exiting due to is_alive()
        while True:
            if not self._render_thread.is_alive():
                break
            start_time = time.time()
            data = self._recv_data()
            self._data_queue.put(data)
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            counter = (counter + 1) % MAXLEN
            if counter == MAXLEN-1:
                avg_fps = len(frame_times) / sum(frame_times)
                self._log(f"Recv FPS: {avg_fps}")
    
    def render_loop(self):
        cv2.namedWindow("Feed Video", cv2.WINDOW_NORMAL)
        time.sleep(0.1) # To prevent exiting due to is_alive()
        while True:
            if not self._receive_thread.is_alive():
                break
            data = self._data_queue.get()
            cv2.imshow("Feed Video", data)
            cv2.waitKey(1)
            
    def wait(self):
        self._receive_thread.join()
        self._render_thread.join()


if __name__ == "__main__":
    server = CarSimListener()

    try:
        server.await_connection()
    except KeyboardInterrupt:
        print("[Main] Exiting...")
        pass
    finally:
        print("[Main] Cleanup...")
        del server
        cv2.destroyAllWindows()