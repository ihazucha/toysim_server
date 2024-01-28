#!/usr/bin/env python3

import socket
from queue import Queue
from threading import Event

from RoboSim.render import RendererThread
from RoboSim.settings import (CAMERA_FRAME_SIZE, SERVER_HOST, SERVER_PORT)


class TCPListener:
    def __init__(self, host:str=SERVER_HOST, port:int=SERVER_PORT, verbose:bool=True):
        self._verbose = verbose
        self._listener_socket = None
        self._client_connection = None
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
            socket, addr = self._sock.accept()
            self._log(f"Connected: {addr}")
            data_queue = Queue(maxsize=2)
            exit_event = Event()
            connection = TCPConnection(socket, data_queue, exit_event, verbose=self._verbose)
            renderer = RendererThread(data_queue, exit_event, verbose=self._verbose)
            renderer.start()
            connection.receive_loop()
            renderer.join()
            self._log(f"Disconnected: {addr}")
        
class TCPConnection:
    def __init__(self, socket, data_queue, exit_event, verbose=True):
        self._socket = socket
        self._data_queue = data_queue
        self._exit_event = exit_event
        self._verbose = verbose
              
    def __del__(self):
        self._close()

    def _log(self, msg:str, who:str="[Conn]"):
        if self._verbose: print(f"{who} {msg}")

    def _close(self):
        if self._socket is None:
            return
        try:
            self._log(f"Closing cocket...")
            self._socket.shutdown(socket.SHUT_RDWR)
            self._socket.close()
            self._log(f"Socket closed")
        except OSError as e:
            self._log(f"Error while closing socket: {e}")
            pass

    def _recv_all(self, size):
        data = b''
        while len(data) < size:
            more = self._socket.recv(size - len(data))
            if not more:
                raise IOError('Socket closed before all data received')
            data += more
        return data
        
    def _recv_data(self):
        return self._recv_all(CAMERA_FRAME_SIZE)

    def receive_loop(self):
        while True:
            try:
                data = self._recv_data()
                self._data_queue.put(data)
            except OSError:
                self._log("Connection closed by client")
                break
        self._exit_event.set()


# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    server = TCPListener()
    try:
        server.await_connection()
    except KeyboardInterrupt:
        print("[Main] Exiting...")