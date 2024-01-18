import numpy as np
import socket
import numpy as np
import cv2

from pprint import pprint

HOST = "127.0.0.1"
PORT = 4502
# PACKET_SIZE = 1048576
PACKET_SIZE = 3145728

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        try:
            while True:
                
                data = conn.recv(PACKET_SIZE)
                
                #verifico che ci sia un pacchetto da gestire
                if not data:
                    print("No data")
                    break

                
                if ( len(data) == PACKET_SIZE ):
                    #gestisco immagine
                    immagine = np.frombuffer(data,dtype=np.uint8)
                    # Transform the array to an RGB image with 1024 x 1024 pixels
                    immagine = immagine.reshape(1024,1024,3)

                    cv2.imshow("Feed Video",immagine)
                    pprint(immagine)
                else:
                    print("Data length: {}".format(len(data)))
                #distruggo tutto se premo Q
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        except KeyboardInterrupt:
            exit()