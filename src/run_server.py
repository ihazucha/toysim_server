#!/usr/bin/env python3

import sys
from argparse import ArgumentParser

from modules.network import NetworkServer, get_local_ip
from modules.processor import Processor
from modules.render import Renderer
from utils.ipc import SPMCQueue


def parse_args():
    parser = ArgumentParser(description="Runs ToySim UI")
    parser.add_argument("-i", "--ip", type=str, default=get_local_ip(), help="Addr where UI (server) listens")
    return parser.parse_args()


def main():
    args = parse_args()
    q_image = SPMCQueue(port=10001)
    q_sensor = SPMCQueue(port=10002)
    q_remote = SPMCQueue(port=10003)

    p_network = NetworkServer(q_image=q_image, q_sensor=q_sensor, q_remote=q_remote, server_ip=args.ip)
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
