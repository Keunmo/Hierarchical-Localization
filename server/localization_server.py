import os
import shutil
import server.Config as Config
import socket
import logging
import math
from _thread import *

import pycolmap
from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

logger = logging.getLogger("hloc")
logger.setLevel(logging.CRITICAL)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

HERE_PATH = os.path.normpath(os.path.dirname(__file__))

target = 'cluster_4f'

images = Path('../datasets/' + target)
outputs = Path('../outputs/' + target)

sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

model = pycolmap.Reconstruction()
model.read_binary(sfm_dir.as_posix())


def quaternion_to_euler(qw, qx, qy, qz):
    t0 = 2.0 * (qw * qx + qy * qz)
    t1 = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll_x = math.atan2(t0, t1) # radian
    roll_x = math.degrees(roll_x) # degree
    
    t2 = 2.0 * (qw * qy - qz * qx)
    t2 = 1.0 if t2 >= 1.0 else t2
    t2 = -1.0 if t2 <= -1.0 else t2
    pitch_y = math.asin(t2) # radian
    pitch_y = math.degrees(pitch_y) # degree
    
    t3 = 2.0 * (qw * qz + qx * qy)
    t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw_z = math.atan2(t3, t4) # radian
    yaw_z = math.degrees(yaw_z) # degree
    # print("Roll(X), Pitch(Y), Yaw(Z):\n", roll_x, pitch_y, yaw_z)
    return roll_x, pitch_y, yaw_z # in degrees
    

def localizer(query): # query img path
    # target = 'cluster_4f'

    # images = Path('../datasets/' + target)
    # outputs = Path('../outputs/' + target)

    # sfm_pairs = outputs / 'pairs-sfm.txt'
    # loc_pairs = outputs / 'pairs-loc.txt'
    # sfm_dir = outputs / 'sfm'
    # features = outputs / 'features.h5'
    # matches = outputs / 'matches.h5'

    # feature_conf = extract_features.confs['superpoint_aachen']
    # matcher_conf = match_features.confs['superglue']

    references = [p.relative_to(images).as_posix() for p in (images / 'mapping/').iterdir()]

    extract_features.main(feature_conf, images, image_list=[query], feature_path=features, overwrite=True)
    pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references)
    match_features.main(matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True)

    # model = pycolmap.Reconstruction()
    # model.read_binary(sfm_dir.as_posix())

    camera = pycolmap.infer_camera_from_image(images / query)
    ref_ids = [model.find_image_with_name(r).image_id for r in references]
    conf = {
        'estimation': {'ransac': {'max_error': 12}},
        'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
    }
    localizer = QueryLocalizer(model, conf)
    ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)

    # print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
    print(ret['qvec'])
    print(ret['tvec'])
    qw, qx, qy, qz = ret['qvec'].tolist()
    tx, ty, tz = ret['tvec'].tolist()

    # roll, pitch, yaw = quaternion_to_euler(qw, qx, qy, qz)
    # res = ' '.join(str(i) for i in [-roll, pitch, -yaw, tx, -ty, tz])

    # # location = ' '.join(str(q) for q in ret['qvec'].tolist())+' '+' '.join(str(t) for t in ret['tvec'].tolist())
    # return res
    return qw, qx, qy, qz, tx, ty, tz

def threaded(client_socket, addr):
    # print('Connected by :', addr[0], ':', addr[1])
    while True:
        try : 
            data = client_socket.recv(4096)
            if not data:
                # print('Disconnected by' + addr[0], ':', addr[1])
                break
            # print('Received from ' + addr[0], ':', addr[1])

            request_data = data.decode('utf-8').split() # cp949
            id = request_data[0]
            file_name = request_data[1]
            file_size = int(request_data[2])
            # print('req data:', request_data)

            dir_path = os.path.join(os.getcwd(), 'client_data', id)
            img_path = os.path.join(dir_path, file_name)
            os.makedirs(dir_path, exist_ok=True)
                
            with open(img_path, 'wb') as f:
                data = client_socket.recv(4096)
                # client_socket.sendall('test_send0'.encode())
                pre = data[-3:]
                while data:
                    f.write(data)
                    data = client_socket.recv(4096)
                    if b'EOF' in pre+data[-3:]:
                        f.write(data[:-3])
                        # print('EOF\n')
                        break
                    # print("length : ", + len(data))
                    pre = data[-3:]
            # print('End write')

            qw, qx, qy, qz, tx, ty, tz = localizer(img_path)
            roll, pitch, yaw = quaternion_to_euler(qw, qx, qy, qz)
            res = ' '.join(str(i) for i in [-roll, pitch+90, -yaw, tx, -ty, tz])

            # client_socket.sendall(localizer(img_path).encode())
            client_socket.sendall(res.encode())

            # print("trajectory transferred")
            
        except ConnectionResetError as e:
            # print('Disconnected by' + addr[0], ':', addr[1])
            break
        except UnicodeDecodeError:
            # print('Unicode Decode error')
            client_socket.sendall("error".encode())
            break
            
    client_socket.close()


HOST = Config.serv_addr
PORT = Config.serv_port

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()

print('server start')

while True:
    print('wait')
    client_socket, addr = server_socket.accept()
    start_new_thread(threaded, (client_socket, addr))

# server_socket.close()