import socket

# Configuration
SERVER_IP = "192.168.0.104"  # Replace with the server's IP address
SERVER_PORT = 10001       # Same port as the server

def main():
    # Create a TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to the server
        sock.connect((SERVER_IP, SERVER_PORT))

        # Open a file to write the received data
        with open("received_video.h264", "wb") as f:
            print("Connected to server. Receiving data...")
            while True:
                # Receive data in chunks
                data = sock.recv(4096)  # Adjust buffer size as needed
                if not data:
                    break  # Break the loop if no more data is received
                f.write(data)  # Write the received data to file

        print("Data received and saved to 'received_video.h264'.")

if __name__ == "__main__":
    main()