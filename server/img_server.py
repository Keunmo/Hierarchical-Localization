import socket
from pathlib import Path
import time
import struct
import pickle
import pycolmap
# from localization_server import localizer
import sys
import os
import srv_conf

now_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(now_dir)
sys.path.append(root_dir)

from hloc import localize_sfm, extract_features, match_features, pairs_from_retrieval


dbname = 'library'

images = Path(f'{root_dir}/datasets/{dbname}')
outputs = Path(f'{root_dir}/outputs/{dbname}')
sfm_dir = outputs / 'sfm'
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']
feature_path = outputs / f'{feature_conf["output"]}.h5'
match_path = outputs / f'{feature_conf["output"]}_{matcher_conf["output"]}_{sfm_pairs.stem}.h5'

model = pycolmap.Reconstruction()
model.read_binary(sfm_dir.as_posix())


# return localize result of given image. 
# must follow below file structure:
# datasets
#     {dbname}
#         mapping
#         {uid}
def localizer(filename: Path, uid: str):
    queries = [(images/uid/filename).relative_to(images).as_posix()]
    extract_features.main(feature_conf, images, image_list=queries, feature_path=feature_path)
    global_descriptors = extract_features.main(retrieval_conf, images, outputs, image_list=queries)
    pairs_from_retrieval.main(global_descriptors, loc_pairs, num_matched=10, db_prefix="mapping", query_prefix=uid)
    match_features.main(matcher_conf, loc_pairs, features=feature_path, matches=match_path)
    query = f"{uid}/{filename}"
    result, _ = localize_sfm.localize_from_image(model, images, query, loc_pairs, feature_path, match_path)
    qvec = result[f'{query}'][0]
    tvec = result[f'{query}'][1]
    # concat tvec and qvec
    result_str = f"{tvec[0]} {tvec[1]} {tvec[2]} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]}"
    print(result_str)
    return result_str


host = srv_conf.host                        
port = srv_conf.port

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

    filename = f"{sent_time}.jpg"
    save_path = images/uid
    save_path.mkdir(parents=True, exist_ok=True)

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
    loc_res = localizer(filename, uid)
    clientsocket.sendall(loc_res.encode())

        
clientsocket.close()
sock.close()