#!/usr/bin/env python3

import sys
from queue import Queue
from threading import Event

from ToySim.server import TcpServer
from ToySim.processor import Processor
from ToySim.render import Renderer

# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    recv_queue = Queue(maxsize=2)
    send_queue = Queue(maxsize=2)
    render_queue = Queue(maxsize=2)
    connected_event = Event()

    server = TcpServer(recv_queue, send_queue, connected_event)
    server.start()

    processor = Processor(recv_queue, send_queue, render_queue, connected_event)
    processor.start()

    renderer = Renderer(render_queue)
    exit_code = renderer.run()
    
    sys.exit(exit_code)
