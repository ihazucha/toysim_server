#!/usr/bin/env python3

# Client consists of 2 threads
# 1. Main (Game) thread which runs the simulation, forwards data to the collecting threads
# 2. Data gathering, processing and sending over the network thread

# The architecture inside the UE should be as follows:
# 1. Game thread handles the simulation and everything that happens inside the game
# 2. The network client should work such that:
#   - on GameStart event, client attempts to connect to the server and receive initialization data
#       - if connection is successful and init data is received, setup the simulation accordingly and spawn a data processing thread
#           - the thread will collect sim data and send them over network to the client
#       - if connection fails, the message is displayed and simulation is offline and game reacts accordingly


import threading
import queue
import numpy as np
import socket
import os
from math import floor

SENDER_HOST = 'localhost'
SENDER_PORT = 6666
SENDER_ADDR = (SENDER_HOST, SENDER_PORT)

RECEIVER_HOST = 'localhost'
RECEIVER_PORT = 3333
RECEIVER_ADDR = (RECEIVER_HOST, RECEIVER_PORT)

DATA_QUEUE = queue.Queue(maxsize=1)

DFRAME_PAYLOAD_SIZE = 3145728
DFRAME_TOTAL_SIZE   = DFRAME_PAYLOAD_SIZE
DFRAME_HEADER_SIZE  = 8
UDP_BUFFER_SIZE     = 65000

import time

def tcp_data_sender():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(RECEIVER_ADDR)
        # settings = s.recv(8)
        # print(f"[Data] Received settings: {settings}")
        while True:
            data = DATA_QUEUE.get()
            s.sendall(data.size.to_bytes(4, 'little'))
            s.sendall(data)



class DummyImageData:
    def __init__(self, w:int, h:int):
        self._counter = 0
    def __call__(self):
        data = np.zeros((1024, 1024, 3), dtype=np.uint8)
        data[self._counter:self._counter+10, :, 1] = 255
        self._counter = (self._counter + 2) % 1024
        return data.flatten()

def game_thread():
    data = DummyImageData(w=1024, h=1024)
    while True:
        DATA_QUEUE.put(data(), block=True)

def data_thread():
    try:
        tcp_data_sender()
        # udp_data_sender()
    except Exception as e:
        print(f"[Data] Exception occured: {e}")
        os._exit(1)

def udp_data_sender():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(SENDER_ADDR)
        
        frame_num:int = 0
        while True:
            data = DATA_QUEUE.get()
            chunk_num:int = 0
            for i in range(0, len(data), UDP_BUFFER_SIZE):
                chunk = data[i:i+UDP_BUFFER_SIZE]
                frame_num_b = frame_num.to_bytes(4, 'little')
                chunk_num_b = chunk_num.to_bytes(4, 'little')
                chunk_b = frame_num_b + chunk_num_b + chunk.tobytes().ljust(UDP_BUFFER_SIZE, b'\0')
                print(f"F:{frame_num}|C:{chunk_num}")
                s.sendto(chunk_b, RECEIVER_ADDR)
                chunk_num += 1
            frame_num += 1

data_t = threading.Thread(target=data_thread)
data_t.daemon = True

try:
    data_t.start()
    while True:
        game_thread()
except KeyboardInterrupt:
    print("[Client] Interrupt request - exiting program")
finally:
    pass