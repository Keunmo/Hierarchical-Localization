import socket
from pathlib import Path
import time
import struct
import pickle


host = 'localhost'                        
port = 55555
save_path = Path("received_img")

# create a socket object
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# bind to the port
sock.bind((host, port))

# queue up to 5 requests
sock.listen(5)                                           

while True:
    # establish a connection
    clientsocket,addr = sock.accept()
    print(f"Got a connection from sock {str(addr)}")

    data = clientsocket.recv(1024)
    uid, sent_time = data.split(b":")
    uid = uid.decode()
    sent_time = sent_time.decode()

    filename = f"{uid}_{sent_time}.jpg"
    
    data_buffer = b""
    data_size = struct.calcsize("L")

    while len(data_buffer) < data_size:
        data_buffer += clientsocket.recv(4096)

    packed_data_size = data_buffer[:data_size]
    data_buffer = data_buffer[data_size:]

    frame_size = struct.unpack("L", packed_data_size)[0]

    while len(data_buffer) < frame_size:
        data_buffer += clientsocket.recv(4096)

    frame_data = data_buffer[:frame_size]
    data_buffer = data_buffer[frame_size:]
    print("수신 프레임 크기 : {} bytes".format(frame_size))

    if frame_size == len(frame_data):
        print("Success")
        clientsocket.sendall(str(frame_size).encode())

    frame = pickle.loads(frame_data)
    with open(save_path/filename, "wb") as f:
        f.write(frame)
    print("dbg: saved image: ", filename)
        
clientsocket.close()
sock.close()