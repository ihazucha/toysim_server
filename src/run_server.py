import sys
from argparse import ArgumentParser

from datalink.network import NetworkServer, TcpServer, get_local_ip
from modules.processor import Processor, ControllerType
from modules.render import Renderer


def parse_args():
    parser = ArgumentParser(description="Runs ToySim UI")
    parser.add_argument("-i", "--ip", type=str, default=get_local_ip(), help="Bind address")
    parser.add_argument("-c", "--controller", type=str, choices=[c.value for c in ControllerType])
    return parser.parse_args()


def main():
    args = parse_args()

    p_network = NetworkServer(server_ip=args.ip)
    p_sim_network = TcpServer(server_ip=args.ip)
    p_processor = Processor(controller_type=ControllerType(args.controller))
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
