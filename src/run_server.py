import sys
from argparse import ArgumentParser

from datalink.network import TcpServer, get_local_ip
from modules.processor import Processor, ControllerType
from modules.render import Renderer
from modules.messaging import messaging as msg


def parse_args():
    parser = ArgumentParser(description="Runs ToySim UI")
    parser.add_argument("-i", "--ip", type=str, default=get_local_ip(), help="Server IP")
    parser.add_argument("--sim_port", type=int, default=8888, help="Server port for simulation")
    parser.add_argument("--real_port", type=int, default=9999, help="Server port for real vehicle")
    parser.add_argument("-c", "--controller", type=str, choices=[c.value for c in ControllerType])
    return parser.parse_args()


def main():
    args = parse_args()

    sim_addr = (args.ip, args.sim_port)
    real_addr = (args.ip, args.real_port)

    p_sim_server = TcpServer(addr=sim_addr, q_recv=msg.q_sim, q_send=msg.q_control, id="sim")
    p_real_server = TcpServer(addr=real_addr, q_recv=msg.q_real, q_send=msg.q_control, id="real")
    p_processor = Processor(controller_type=ControllerType(args.controller))
    renderer = Renderer()

    processes = [
        p_sim_server,
        p_real_server,
        p_processor,
    ]

    [p.start() for p in processes]
    exit_code = renderer.run()
    [p.terminate() for p in processes]
    [p.join() for p in processes]

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
