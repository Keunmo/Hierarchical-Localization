# import socket
# import time
from pathlib import Path

# server_host = 'localhost'
# server_port = 55555
# img_path = Path("img_data/image.jpg")

# # create a socket object
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# # get local machine name

# # connection to hostname on the port.
# sock.connect((server_host, server_port))

# uid = "asdf1234"
# sent_time = str(int(time.time()))
# sock.sendall(f"{uid}:{sent_time}".encode())

# with open(img_path, "rb") as f:
#     data = f.read()
#     sock.sendall(data)
    
# length = sock.recv(1024)
# print("Received image length: ", length.decode())

# sock.close()

# 필요한 패키지 import
import socket # 소켓 프로그래밍에 필요한 API를 제공하는 모듈
import pickle # ﻿객체의 직렬화 및 역직렬화 지원 모듈﻿
import struct # 바이트(bytes) 형식의 데이터 처리 모듈

# 서버 ip 주소 및 port 번호
ip = 'localhost'
port = 55555
img_path = Path("img_data/image.jpg")

# 소켓 객체 생성
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    # 서버와 연결
    client_socket.connect((ip, port))
    
    print("연결 성공")
    
    with open(img_path, "rb") as f:
        data = f.read()
        data = pickle.dumps(data)
        print("전송 프레임 크기 : {} bytes".format(len(data)))
        client_socket.sendall(struct.pack("L", len(data)) + data)
