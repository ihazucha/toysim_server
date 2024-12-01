#!/usr/bin/env python3

import sys
import argparse

from multiprocessing import Manager, Queue, Event

from ToySim.server import TcpServer, Network
from ToySim.processor import Processor
from ToySim.render import Renderer
from ToySim.settings import ClientTypes

# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ToySim server.")
    parser.add_argument("-c", "--client", choices=["sim", "veh"], required=True, help="Client type")

    args = parser.parse_args()
    client_map = {"sim": ClientTypes.SIMULATION, "veh": ClientTypes.VEHICLE}
    client = client_map[args.client]

    q_image: Queue = Queue(1)
    q_sensor: Queue = Queue(1)
    q_control: Queue = Queue(1)
    q_input: Queue = Queue(1)
    e_connected = Event()

    network = Network(q_image=q_image, q_sensor=q_sensor, q_control=q_control)
    network.start()

    processor = Processor(q_recv, q_send, q_render, e_connected, client=client)
    processor.start()

    renderer = Renderer(render_queue, client=client)
    exit_code = renderer.run()

    sys.exit(exit_code)


class MultiReaderQueue:
    """One producer, multiple (possibly lagging a frame or two behind) consumers"""
    def __init__(self):
        self._manager = Manager()