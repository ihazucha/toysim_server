#!/usr/bin/env python3

import sys


from ToySim.modules.network import NetworkServer
from ToySim.modules.processor import Processor
from ToySim.modules.render import Renderer
from ToySim.utils.ipc import SPMCQueue


def main():
    q_image = SPMCQueue(port=10001)
    q_sensor = SPMCQueue(port=10002)
    q_remote = SPMCQueue(port=10003)

    p_network = NetworkServer(q_image=q_image, q_sensor=q_sensor, q_remote=q_remote)
    p_processor = Processor(q_image=q_image, q_sensor=q_sensor, q_remote=q_remote)
    renderer = Renderer(q_image=q_image, q_sensor=q_sensor, q_remote=q_remote)

    processes = [p_network, p_processor]

    [p.start() for p in processes]
    exit_code = renderer.run()
    [p.terminate() for p in processes]
    [p.join() for p in processes]

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
