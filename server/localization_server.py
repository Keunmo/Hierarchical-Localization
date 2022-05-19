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

# target = 'jaram220511'

# images = Path('../datasets/' + target)
# outputs = Path('../outputs/' + target)

# sfm_pairs = outputs / 'pairs-sfm.txt'
# loc_pairs = outputs / 'pairs-loc.txt'
# sfm_dir = outputs / 'sfm'
# features = outputs / 'features.h5'
# matches = outputs / 'matches.h5'

# feature_conf = extract_features.confs['superpoint_aachen']
# matcher_conf = match_features.confs['superglue']

# model = pycolmap.Reconstruction()
# model.read_binary(sfm_dir.as_posix())

# references = [p.relative_to(images).as_posix() for p in (images / 'mapping/').iterdir()]
# ref_ids = [model.find_image_with_name(r).image_id for r in references]

def localizer(map_name: str, query: str): # query img path
    # target = 'jaram220511'
    target = map_name

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

    references = [p.relative_to(images).as_posix() for p in (images / 'mapping/').iterdir()]
    ref_ids = [model.find_image_with_name(r).image_id for r in references]
    # references = [p.relative_to(images).as_posix() for p in (images / 'mapping/').iterdir()]
    extract_features.main(feature_conf, images, image_list=[query], feature_path=features, overwrite=True)
    pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references)
    match_features.main(matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True)

    camera = pycolmap.infer_camera_from_image(images / query)
    # ref_ids = [model.find_image_with_name(r).image_id for r in references]
    conf = {
        'estimation': {'ransac': {'max_error': 12}},
        'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
    }
    localizer = QueryLocalizer(model, conf)
    ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)

    print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
    print(ret['qvec'])
    print(ret['tvec'])
    qw, qx, qy, qz = ret['qvec'].tolist()
    tx, ty, tz = ret['tvec'].tolist()

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
            print('req data:', request_data)
            id = request_data[0]
            map_name = request_data[1]
            file_name = request_data[2]
            file_size = int(request_data[3])

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

            # qw, qx, qy, qz, tx, ty, tz = localizer(img_path)
            trans = ' '.join(str(i) for i in localizer(map_name, img_path))

            # client_socket.sendall(localizer(img_path).encode())
            client_socket.sendall(trans.encode())

            # print("trajectory transferred")
            
        # except ConnectionResetError as e:
        #     # print('Disconnected by' + addr[0], ':', addr[1])
        #     break
        except Exception as e:
            print('Error:', e)
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