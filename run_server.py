#!/usr/bin/env python3

import sys
import os


from ToySim.server import Network
from ToySim.processor import Processor
from ToySim.render import Renderer
from ToySim.ipc import SharedBuffer

DEBUG = os.getenv("DEBUG", 0)

def pdebug(msg: str):
    if DEBUG:
        print(msg)

def main():
    q_image = SharedBuffer(2)
    q_sensor = SharedBuffer(2)
    q_control = SharedBuffer(2)

    pdebug("test")

    p_network = Network(
        q_image=q_image.get_writer(),
        q_sensor=q_sensor.get_writer(),
        q_control=q_control.get_reader(),
    )

    p_processor = Processor(
        q_image=q_image.get_reader(),
        q_sensor=q_sensor.get_reader(),
        q_control=q_control.get_writer(),
    )

    renderer = Renderer(
        q_image=q_image.get_reader(),
        q_sensor=q_sensor.get_reader(),
        q_control=q_control.get_reader(),
    )

    processes = [p_network, p_processor]

    [p.start() for p in processes]

    exit_code = renderer.run()
    [p.join() for p in processes]

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
