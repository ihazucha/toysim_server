#!/usr/bin/env python3

from math import floor
import socket
import struct
import numpy as np
import cv2

from threading import Event
from queue import Queue

from RoboSim.render import ImageDataRenderer


RECEIVER_HOST = 'localhost'
RECEIVER_PORT = 3333
RECEIVER_ADDR = (RECEIVER_HOST, RECEIVER_PORT)

SENDER_HOST = 'localhost'
SENDER_PORT = 6666
SENDER_ADDR = (SENDER_HOST, SENDER_PORT)

DFRAME_PAYLOAD_SIZE = 3145728
DFRAME_TOTAL_SIZE   = DFRAME_PAYLOAD_SIZE
DFRAME_HEADER_SIZE  = 8
UDP_BUFFER_SIZE     = 65000
UDP_TOTAL_SIZE      = DFRAME_HEADER_SIZE + UDP_BUFFER_SIZE

# TODO: possible error when DFRAME_TOTAL_SIZE is divisible by UDP_BUFFER_SIZE
DFRAME_CHUNK_COUNT = DFRAME_TOTAL_SIZE / UDP_BUFFER_SIZE
DFRAME_CHUNK_COUNT_FLOOR = floor(DFRAME_CHUNK_COUNT)
DFRAME_CHUNK_HAS_REMAINDER = DFRAME_CHUNK_COUNT > DFRAME_CHUNK_COUNT_FLOOR
DFRAME_CHUNK_COUNT_FINAL = DFRAME_CHUNK_COUNT_FLOOR + 1 if DFRAME_CHUNK_HAS_REMAINDER else DFRAME_CHUNK_COUNT_FLOOR
DFRAME_PADDED_SIZE  = DFRAME_CHUNK_COUNT_FINAL * UDP_BUFFER_SIZE   
         
class UdpDataReceiver:
    def __init__(self, data_queue:Queue, verbose=True):
        self._verbose = verbose
        self._data_queue = data_queue
        self._exit_event = Event()
        
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(RECEIVER_ADDR)

    @property
    def exit_event(self):
        return self._exit_event

    def __del__(self):
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

    def _log(self, msg:str, who:str="[Receiver]"):
        if self._verbose: print(f"{who} {msg}")
    

    def _parse_data_header(self, header: bytes):
        payload_size, = struct.unpack("i", header)
        return payload_size
        
    def _recv_data(self):
        remaining_payload_size = DFRAME_PADDED_SIZE
        payload:bytes = b''
        while remaining_payload_size > 0:
            chunk, _ = self._sock.recvfrom(UDP_TOTAL_SIZE)
            frame_counter, chunk_counter = struct.unpack("ii", chunk[:8])
            print(f"F:{frame_counter}|C:{chunk_counter}")
            payload += chunk[8:]
            remaining_payload_size -= UDP_BUFFER_SIZE
        return payload[:DFRAME_PAYLOAD_SIZE]

    def loop(self):
        # self._send_settings()
        while True:
            if self._exit_event.is_set():
                break
            try:
                data = self._recv_data()
                self._data_queue.put(data)
            except OSError as e:
                self._log(f"Connection closed by client: {e}")
                self._exit_event.set()
                break


if __name__ == "__main__":
    try:
        data_queue = Queue(maxsize=1)
        server = UdpDataReceiver(data_queue)
        exit_event = server.exit_event
        renderer = ImageDataRenderer(data_queue, exit_event)
        renderer.start()
        server.loop()
    except KeyboardInterrupt:
        print("[Main] Exiting...")
        pass
    finally:
        print("[Main] Cleanup...")
        cv2.destroyAllWindows()