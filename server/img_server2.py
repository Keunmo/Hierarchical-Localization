# import socket
from pathlib import Path

# # create a socket object
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# # get local machine name
# host = 'localhost'                        
# port = 55555
# save_path = Path("received_img")
# # bind to the port
# sock.bind((host, port))

# # queue up to 5 requests
# sock.listen(5)                                           

# while True:
#    # establish a connection
#    clientsocket,addr = sock.accept()      

#    print(f"Got a connection from sock {str(addr)}")
#    data = clientsocket.recv(1024)
#    uid, sent_time = data.split(b":")
#    uid = uid.decode()
#    sent_time = sent_time.decode()
#    filename = "{uid}_{sent_time}.jpg"
#    with open(filename, 'wb') as f:
#         while True:
#             data = clientsocket.recv(1024)
#             if not data:
#                 break
#             # write data to a file
#             f.write(data)
#    clientsocket.send(str(len(data)).encode())
#    clientsocket.close()

# 필요한 패키지 import
import socket # 소켓 프로그래밍에 필요한 API를 제공하는 모듈
import struct # 바이트(bytes) 형식의 데이터 처리 모듈
import pickle # ﻿객체의 직렬화 및 역직렬화 지원 모듈﻿

# 서버 ip 주소 및 port 번호
ip = 'localhost'
port = 55555

# 소켓 객체 생성
server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

# 소켓 주소 정보 할당
server_socket.bind((ip, port))

# 연결 리스닝(동시 접속) 수 설정
server_socket.listen(10) 

print('클라이언트 연결 대기')

# 연결 수락(클라이언트 (소켓, 주소 정보) 반환)
client_socket, address = server_socket.accept()
print('클라이언트 ip 주소 :', address[0])

# 수신한 데이터를 넣을 버퍼(바이트 객체)
data_buffer = b""

# calcsize : 데이터의 크기(byte)
# - L : 부호없는 긴 정수(unsigned long) 4 bytes
data_size = struct.calcsize("L")

while True:
    # 설정한 데이터의 크기보다 버퍼에 저장된 데이터의 크기가 작은 경우
    while len(data_buffer) < data_size:
        # 데이터 수신
        data_buffer += client_socket.recv(4096)

    # 버퍼의 저장된 데이터 분할
    packed_data_size = data_buffer[:data_size]
    data_buffer = data_buffer[data_size:] 
    
    # struct.unpack : 변환된 바이트 객체를 원래의 데이터로 반환
    # - > : 빅 엔디안(big endian)
    #   - 엔디안(endian) : 컴퓨터의 메모리와 같은 1차원의 공간에 여러 개의 연속된 대상을 배열하는 방법
    #   - 빅 엔디안(big endian) : 최상위 바이트부터 차례대로 저장
    # - L : 부호없는 긴 정수(unsigned long) 4 bytes 
    frame_size = struct.unpack("L", packed_data_size)[0]
    
    # 프레임 데이터의 크기보다 버퍼에 저장된 데이터의 크기가 작은 경우
    while len(data_buffer) < frame_size:
        # 데이터 수신
        data_buffer += client_socket.recv(4096)
    
    # 프레임 데이터 분할
    frame_data = data_buffer[:frame_size]
    data_buffer = data_buffer[frame_size:]
    
    print("수신 프레임 크기 : {} bytes".format(frame_size))
    if frame_size == len(frame_data):
        print("Success")
    # loads : 직렬화된 데이터를 역직렬화
    # - 역직렬화(de-serialization) : 직렬화된 파일이나 바이트 객체를 원래의 데이터로 복원하는 것
    frame = pickle.loads(frame_data)
    save_path = Path("received_img")
    with open(f"{save_path}/recv_img.jpg", 'wb') as f:
        f.write(frame)

# 소켓 닫기
client_socket.close()
server_socket.close()
print('연결 종료')

