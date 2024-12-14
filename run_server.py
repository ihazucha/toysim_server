#!/usr/bin/env python3

import sys


from ToySim.server import Network
from ToySim.processor import Processor
from ToySim.render import Renderer
from ToySim.ipc import SPMCQueue


def main():
    q_image = SPMCQueue(port=10001)
    q_sensor = SPMCQueue(port=10002)
    q_control = SPMCQueue(port=10003)

    p_network = Network(q_image=q_image, q_sensor=q_sensor, q_control=q_control)
    p_processor = Processor(q_image=q_image, q_sensor=q_sensor, q_control=q_control)
    renderer = Renderer(q_image=q_image, q_sensor=q_sensor, q_control=q_control)

    processes = [p_network, p_processor]

    [p.start() for p in processes]
    exit_code = renderer.run()
    [p.terminate() for p in processes]
    [p.join() for p in processes]

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
