# multi_sub.py

import zmq
import time
import os

from typing import List
from multiprocessing import Process, Event
from threading import Thread

host = "127.0.0.1"
port = "5007"


def publisher(sub_ready):
    ctx = zmq.Context()
    with ctx.socket(zmq.PUB) as socket:
        # socket.bind(f"tcp://{host}:{port}")
        socket.bind(f"ipc:///tmp/pubsubtest")
        sub_ready.wait()
        while True:
            str = "1" * 20000
            # str = "light is ON"
            socket.send_string(str)


def subscriber(sub_ready):
    def thread_loop(appliance):
        ctx = zmq.Context()
        with ctx.socket(zmq.SUB) as socket:
            # socket.connect(f"tcp://{host}:{port}")
            socket.connect(f"ipc:///tmp/pubsubtest")
            socket.subscribe("")
            while True:
                light_msg = socket.recv_string()
                print(f"[{appliance}] received '{len(light_msg)}' from light.")

    t_coffee = Thread(target=thread_loop, args=("Coffee",))
    t_toaster = Thread(target=thread_loop, args=("Toaster",))

    ts = [t_coffee, t_toaster]
    [t.start() for t in ts]
    sub_ready.set()
    [t.join() for t in ts]

# def subscriber(sub_ready):
#     ctx = zmq.Context()
#     socket = ctx.socket(zmq.SUB)
#     socket.connect(f"tcp://{host}:{port}")
#     socket.subscribe("light")
#     sub_ready.set()
#     while True:
#         light_msg = socket.recv_string()
#         print(f"received '{light_msg}' from light.")


if __name__ == "__main__":
    e_sub_ready = Event()
    sub = Process(target=subscriber, args=[e_sub_ready])
    pub = Process(target=publisher, args=[e_sub_ready])

    ps: List[Process] = [sub, pub]

    try:
        [p.start() for p in ps]
        [p.join() for p in ps]
    except (KeyboardInterrupt, SystemExit):
        [p.terminate() for p in ps]
    finally:
        [p.join() for p in ps]
        os._exit(0)
