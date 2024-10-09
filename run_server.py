#!/usr/bin/env python3

import sys
import argparse
from queue import Queue
from threading import Event

from ToySim.server import TcpServer
from ToySim.processor import Processor
from ToySim.render import Renderer
from ToySim.settings import ClientTypes

# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ToySim server.")
    parser.add_argument(
        '-c', '--client', 
        choices=['sim', 'veh'], 
        required=True, 
        help="Specify the client type: 'sim' or 'veh'."
    )
    args = parser.parse_args()
    client_map = {'sim': ClientTypes.SIMULATION, 'veh': ClientTypes.VEHICLE}
    client = client_map[args.client]

    recv_queue = Queue(maxsize=2)
    send_queue = Queue(maxsize=1)
    render_queue = Queue(maxsize=2)
    connected_event = Event()

    server = TcpServer(recv_queue, send_queue, connected_event, client=client)
    server.start()

    processor = Processor(recv_queue, send_queue, render_queue, connected_event, client=client)
    processor.start()

    renderer = Renderer(render_queue, client=client)
    exit_code = renderer.run()
    
    sys.exit(exit_code)
