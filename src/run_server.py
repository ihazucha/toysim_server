#!/usr/bin/env python3

import sys
from argparse import ArgumentParser

from modules.network import NetworkServer, TcpServer, get_local_ip
from modules.processor import Processor
from modules.render import Renderer

def parse_args():
    parser = ArgumentParser(description="Runs ToySim UI")
    parser.add_argument("-i", "--ip", type=str, default=get_local_ip(), help="Bind address")
    parser.add_argument("-c", "--controller", help="[dualsense|redroadmarks]")
    return parser.parse_args()


def main():
    args = parse_args()

    p_network = NetworkServer(server_ip=args.ip)
    p_sim_network = TcpServer(server_ip=args.ip)
    p_processor = Processor(controller=args.controller)
    renderer = Renderer()

    processes = [
        p_network,
        p_sim_network,
        p_processor,
    ]

    [p.start() for p in processes]
    exit_code = renderer.run()
    [p.terminate() for p in processes]
    [p.join() for p in processes]

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
