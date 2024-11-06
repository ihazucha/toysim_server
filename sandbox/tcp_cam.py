import sys
import cv2
import numpy as np
import argparse
import socket
import struct
import traceback

MAX_DGRAM = 2**16

parser = argparse.ArgumentParser(description='')  
parser.add_argument('-a', '--addr', type=str, default='localhost', help='Bind port')  
parser.add_argument('-p', '--port', type=int, default='5555', help='Bind address')  
args = parser.parse_args()

class UDPImageServer:
    MAX_DGRAM_SIZE = 2**16

    def __init__(self, addr:str, port:int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((addr, port)) 

    def recv_image(self) -> tuple[bytes, int]:
        image = b''
        segment = 255
        while segment > 1:
            dgram = self._sock.recv(self.__class__.MAX_DGRAM_SIZE)
            segment, timestamp = struct.unpack('=BQ', dgram[0:9])
            image += dgram[9:]
        return (image, timestamp)

try:
    udp = UDPImageServer(addr=args.addr, port=args.port)
    prev_timestamp = 0
    while True:
        (jpg_data, timestamp) = udp.recv_image()
        try:
            image = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR)
        except ValueError as e:
            print(f'[Server] JPG decode problem: {e}')
            continue
        text = f'dt: {timestamp - prev_timestamp} [ms]'
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        prev_timestamp = timestamp
        cv2.imshow('UDPCam', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
except (KeyboardInterrupt, SystemExit):
    pass
finally:
    cv2.destroyAllWindows()         # closes the windows opened by cv2.imshow()
    sys.exit()