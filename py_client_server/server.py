#!/usr/bin/env python3

# Server can actually only consist of one process since it's UDP connection, and only filter out valid packets
# Loop looks as follows:
#   1. Expect first message from client and block until it's received
#   2. Respond by sending settings
#   3. Expect confirmation of settings
#   4. Expect data consisting of:
#       Data struct of fixed size which consists of:
#       control_flag: uint8 where 1 means data stream continues, 0 means data stream ended
#      If no data arrived in TIMEOUT seconds, reset client  

import socket
import struct

import numpy as np
import cv2

import time
import collections 

HOST = 'localhost'
PORT = 6666


class CarSimServer:
    HELLO_SIZE          = 5
    DATA_HEADER_SIZE    = 4
    CAMERA_DATA_SIZE    = 3145728
    
    def __init__(self, host=HOST, port=PORT, verbose=True):
        self._client_connected = False
        self._client_addr = None
        self._verbose = verbose
        
        self._sock = None
        self._conn = None
        self._open_socket(host, port)
       
    def __del__(self):
        self._close_socket()
      
    def _open_socket(self, host, port):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        addr = (host, port)
        self._sock.bind(addr)
        self._sock.listen()
        if self._verbose: print(f"[Server] Listening on {addr}")
    
    def _close_socket(self):
        if self._sock is None:
            return
        try:
            if self._verbose: print(f"[Server] Closing socket...")
            self._sock.shutdown(socket.SHUT_RDWR)
            self._sock.close()
            if self._verbose: print(f"[Server] Socket closed")
        except OSError:
            pass
        
    def _recv_hello(self):
        if self._verbose: print(f"[Server] Waiting for client...")
        
        self._conn, addr = self._sock.accept()
        data = self._recv_all(CarSimServer.HELLO_SIZE)
        if data != b"hello":
            if self._verbose: print(f"[Server] Invalid client hello from: {addr}")
            return False
        self._client_connected = True
        self._client_addr = addr
        if self._verbose: print(f"[Server] Client connected: {addr}")
        return self._client_connected
    
    def _send_settings(self):
        if self._verbose: print(f"[Server] Sending settings...")
        sent = self._conn.sendall(b"settings")
        if self._verbose: print(f"[Server] Settings sent ({sent} B)")
    
    def _recv_data(self):
        header = self._recv_all(CarSimServer.DATA_HEADER_SIZE)
        payload_size, = self._parse_data_header(header)
        if payload_size == 0:
            return None
        payload = self._recv_all(payload_size)
        payload_parsed = self._parse_data_payload(payload)
        # if self._verbose: print("[Server] Recved data of {} B".format(CarSimServer.CAMERA_DATA_SIZE))
        return payload_parsed
    
    def _recv_all(self, size):
        data = b''
        while len(data) < size:
            more = self._conn.recv(size - len(data))
            if not more:
                raise IOError('Socket closed before all data received')
            data += more
        return data
    
    def _parse_data_header(self, header: bytes):
        return struct.unpack("i", header)
        
    def _parse_data_payload(self, payload: bytes):
        # 8-bit 1024x1024 image
        image = np.frombuffer(payload, dtype=np.uint8)
        image = image.reshape((1024, 1024, 3))
        return image
    
    def loop(self, target:callable=None):
        self._recv_hello()
        if not self._client_connected:
            return
        self._send_settings()
        
        frame_times = collections.deque(maxlen=120)      
        while True:
            start_time = time.time()
            data = self._recv_data()
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            avg_fps = len(frame_times) / sum(frame_times)
            print(avg_fps)
            if target:
                target(data)
  
  

if __name__ == "__main__":
    cv2.namedWindow("Feed Video", cv2.WINDOW_NORMAL)
    server = CarSimServer()

    def render_image(data):
        cv2.imshow("Feed Video",data)
        cv2.waitKey(1)

    

    try:
        server.loop(target=render_image)
    except KeyboardInterrupt:
        print("[Main] Exiting...")
        pass
    finally:
        print("[Main] Cleanup...")
        del server
        cv2.destroyAllWindows()