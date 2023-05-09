import socket
import time
import struct # 바이트(bytes) 형식의 데이터 처리 모듈
import pickle
import srv_conf
from pathlib import Path

server_ip = srv_conf.server_ip
server_port = srv_conf.server_port

img_path = Path("img_data/image.jpg")

# create a socket object
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
# connection to hostname on the port.
sock.connect((server_ip, server_port))

uid = "user1234"
sent_time = str(int(time.time()))
sock.sendall(f"{uid}:{sent_time}".encode())

# with open(img_path, "rb") as f:
#     data = f.read()
#     sock.sendall(data)
with open(img_path, "rb") as f:
    data = f.read()
    data = pickle.dumps(data)
    print("전송 프레임 크기 : {} bytes".format(len(data)))
    sock.sendall(struct.pack("L", len(data)) + data)

print("dbg: sent image")
# while True:
#     length = sock.recv(100)
#     print("dbg: received length", length)
#     if not length:
#         break
length = sock.recv(1024)
print("Received image length: ", length.decode())
loc_result = sock.recv(1024)
print("Received localization result: ", loc_result.decode())

sock.close()