#!/usr/bin/env python3

import sys
from argparse import ArgumentParser

from modules.network import NetworkServer, TcpServer, get_local_ip
from modules.processor import Processor
from modules.render import Renderer
from utils.ipc import SPMCQueue


def parse_args():
    parser = ArgumentParser(description="Runs ToySim UI")
    parser.add_argument("-i", "--ip", type=str, default=get_local_ip(), help="Bind address")
    parser.add_argument("-c", "--controller", help="[dualsense|redroadmarks]")
    return parser.parse_args()


def main():
    args = parse_args()
    q_image = SPMCQueue(port=10001)
    q_sensor = SPMCQueue(port=10002)
    q_remote = SPMCQueue(port=10003)
    q_simulation = SPMCQueue(port=10004)

    p_network = NetworkServer(
        q_image=q_image, q_sensor=q_sensor, q_remote=q_remote, server_ip=args.ip
    )
    p_sim_network = TcpServer(q_recv=q_simulation, q_send=q_remote, server_ip=args.ip)
    p_processor = Processor(q_image=q_image, q_sensor=q_sensor, q_remote=q_remote, q_simulation=q_simulation, controller=args.controller)
    renderer = Renderer(q_image=q_image, q_sensor=q_sensor, q_remote=q_remote, q_simulation=q_simulation)

    processes = [p_network, p_sim_network, p_processor]

    [p.start() for p in processes]
    exit_code = renderer.run()
    [p.terminate() for p in processes]
    [p.join() for p in processes]

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
