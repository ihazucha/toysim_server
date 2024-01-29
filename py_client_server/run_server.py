#!/usr/bin/env python3

import sys
from queue import Queue
from threading import Event

from RoboSim.server import TcpServer
from RoboSim.processor import Processor
from RoboSim.render import Renderer

# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    recv_queue = Queue(maxsize=2)
    send_queue = Queue(maxsize=2)
    render_queue = Queue(maxsize=2)
    connected_event = Event()

    bridge = TcpServer(recv_queue, send_queue, connected_event)
    bridge.start()

    processor = Processor(recv_queue, send_queue, render_queue, connected_event)
    processor.start()

    renderer = Renderer(render_queue)
    exit_code = renderer.run()
    
    sys.exit(exit_code)
