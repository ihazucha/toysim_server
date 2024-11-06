#!/usr/bin/env python3

import sys
import argparse
import socket
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
        help="Differentiates between simulation, real vehicle, and potentially other custom configurations"
    )
    parser.add_argument(
        '-ip', '--address',
        required=False,
        type=str,
        help='Listener socket IPv4 address',
        default='localhost'
    )
    parser.add_argument(
        '-p', '--port',
        required=False,
        type=int,
        default='8888',
        help='Listener socket port'
    )
    args = parser.parse_args()
    client_map = {'sim': ClientTypes.SIMULATION, 'veh': ClientTypes.VEHICLE}
    client = client_map[args.client]

    # TODO: One process/thread should never be reading and writing to the same queue
    # to avoid deadlock
    recv_queue = Queue(maxsize=2)
    send_queue = Queue(maxsize=2)
    render_queue = Queue(maxsize=2)
    connected_event = Event()

    server = TcpServer(recv_queue, send_queue, connected_event, address=(args.address, args.port), client=client)
    server.start()

    processor = Processor(recv_queue, send_queue, render_queue, connected_event, client=client)
    processor.start()

    renderer = Renderer(render_queue, client=client)
    exit_code = renderer.run()
    
    sys.exit(exit_code)
