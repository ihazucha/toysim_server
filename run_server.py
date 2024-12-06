#!/usr/bin/env python3

import sys

from ToySim.server import Network
from ToySim.processor import Processor
from ToySim.render import Renderer
from ToySim.utils import SharedBuffer


def main():
    q_image = SharedBuffer(2)
    q_sensor = SharedBuffer(2)
    q_control = SharedBuffer(2)

    network = Network(
        q_image=q_image.get_writer(),
        q_sensor=q_sensor.get_writer(),
        q_control=q_control.get_reader(),
    )

    processor = Processor(
        q_image=q_image.get_reader(),
        q_sensor=q_sensor.get_reader(),
        q_control=q_control.get_writer(),
    )

    renderer = Renderer(
        q_image=q_image.get_reader(),
        q_sensor=q_sensor.get_reader(),
        q_control=q_control.get_reader(),
    )

    exit_code = renderer.run()
    network.start()
    processor.start()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
