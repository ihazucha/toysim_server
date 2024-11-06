"""A test that subscribes to NumPy arrays.

Uses REQ/REP (on PUB/SUB socket + 1) to synchronize
"""

# -----------------------------------------------------------------------------
#  Copyright (c) 2010 Brian Granger
#
#  Distributed under the terms of the New BSD License.  The full license is in
#  the file LICENSE.BSD, distributed as part of this software.
# -----------------------------------------------------------------------------

import sys
import time
import socket
import numpy
import zmq

def parse_args():
    if len(sys.argv) != 4:
        print('usage: subscriber <bind_to> <array-size> <array-count>')
        sys.exit(1)
    args = {}

    try:
        args['bind_to'] = sys.argv[1]
        args['array_size'] = int(sys.argv[2])
        args['array_count'] = int(sys.argv[3])
    except (ValueError, OverflowError) as e:
        print('array-count must be integers')
        sys.exit(1)
    return args


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

def zmq_main():
    args = parse_args()

    ctx = zmq.Context()
    s = ctx.socket(zmq.PULL)
    # s.setsockopt(zmq.RCVBUF, 10 * 1024 * 1024)
    s.bind(args['bind_to'])

    start = time.time()

    print("Receiving arrays...")
    for i in range(args['array_count']):
        print(f'\r{i}', end='')
        # a = s.recv_pyobj()
        a = recv_array(s)
    print("\tDone.")

    end = time.time()

    elapsed = end - start
    print(f'elapsed: {elapsed} [s]')

    throughput = float(args['array_count']) / elapsed

    message_size = len(a)
    megabits = float(throughput * message_size * 8) / 1_000_000

    print(f"message size: {message_size:.0f} [B]")
    print(f"array count: {args['array_count']:.0f}")
    print(f"mean throughput: {throughput:.2f} [msg/s]")
    print(f"mean throughput: {megabits:.3f} [Mb/s]")

    time.sleep(0.2)

def tcp_main():
    args = parse_args()
    print(args)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    addr, port = args['bind_to'].split('//')[1].split(":")
    port = int(port)
    server_socket.bind((addr, port))  # Bind to the specified address and an available port
    server_socket.listen(1)

    print("Waiting for connection...")
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    start = time.time()

    print("Receiving arrays...")
    for i in range(args['array_count']):
        print(f'\r{i}', end='')
        data = conn.recv(args['array_size'])  # Adjust buffer size as needed
        if not data:
            break
    print("\tDone.")

    end = time.time()

    elapsed = end - start
    print(f'elapsed: {elapsed} [s]')

    throughput = float(args['array_count']) / elapsed

    message_size = len(data)
    megabits = float(throughput * message_size * 8) / 1_000_000

    print(f"message size: {message_size:.0f} [B]")
    print(f"array count: {args['array_count']:.0f}")
    print(f"mean throughput: {throughput:.2f} [msg/s]")
    print(f"mean throughput: {megabits:.3f} [Mb/s]")

    conn.close()
    server_socket.close()


if __name__ == "__main__":
    # tcp_main()
    zmq_main()