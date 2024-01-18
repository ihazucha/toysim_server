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
import time
import numpy as np
import socket
import cv2

HOST = 'localhost'
PORT = 6666

DATA_QUEUE = queue.Queue(maxsize=1)
CAMERA_DATA_SIZE    = 3145728


def game_thread():
    while True:
        data = np.random.randint(low=0, high=256, size=CAMERA_DATA_SIZE, dtype=np.uint8)
        DATA_QUEUE.put(data, block=True)

def data_thread():
    addr = (HOST, PORT)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(addr)
        print("[Data] sending hello...")
        s.sendall(b"hello")
        settings = s.recv(8)
        print(f"[Data] Received settings: {settings}")
        while True:
            data = DATA_QUEUE.get()
            print(data.size)
            s.sendall(data.size.to_bytes(4, 'little'))
            s.sendall(data)

        

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