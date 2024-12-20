import socket
import time

if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("localhost", 5500))
        while True:
            data = s.recv(65500)
            print(len(data))
            # time.sleep(0.05)