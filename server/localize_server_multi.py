import os
import shutil
import server.Config as Config
import socket
import logging
import math
from _thread import *

import pycolmap
from pathlib import Path
from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction, visualization, pairs_from_exhaustive
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

logger = logging.getLogger("hloc")
logger.setLevel(logging.DEBUG)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

HERE_PATH = os.path.normpath(os.path.dirname(__file__))

target = 'fastfive'

images = Path('../datasets/' + target)
ref_images = Path(images / 'mapping')
# outputs = Path('../outputs/' + target)
query_images = Path('../datasets/' + target + '/query')
outputs = Path('../outputs/' + 'fastfive-10pair')

sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
# features = outputs / 'features.h5'
# matches = outputs / 'matches.h5'

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']
retrieval_conf = extract_features.confs['netvlad']

features = outputs / f'loc_{feature_conf["output"]}.h5'
gfeatures = outputs / f'loc_{retrieval_conf["output"]}.h5'
matches = outputs / f'loc_{feature_conf["output"]}_{matcher_conf["output"]}.h5'

model = pycolmap.Reconstruction()
model.read_binary(sfm_dir.as_posix())

references = [p.relative_to(images).as_posix() for p in (images / 'mapping').iterdir()]
ref_ids = [model.find_image_with_name(r).image_id for r in references]

conf = {
    'estimation': {'ransac': {'max_error': 12}},
    'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
}
localizer = QueryLocalizer(model, conf)

def localizer(query): # query img path
    # global_descriptors = extract_features.main(retrieval_conf, ref_images, outputs, overwrite=True)
    global_descriptors = extract_features.main(retrieval_conf, images, outputs, image_list=[query])
    pairs_from_retrieval.main(global_descriptors, loc_pairs, num_matched=10, query_list=[query], db_model=sfm_dir)
    features = extract_features.main(feature_conf, query_images, outputs)
    loc_matches = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)

    # extract_features.main(feature_conf, images, image_list=[query], feature_path=features, overwrite=True)
    # pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references)
    # match_features.main(matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True)

    camera = pycolmap.infer_camera_from_image(images / query)

    localizer = QueryLocalizer(model, conf)
    ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)

    print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
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

            res = ' '.join(str(i) for i in localizer(img_path))

            # client_socket.sendall(localizer(img_path).encode())
            client_socket.sendall(res.encode())

            # print("trajectory transferred")

        except Exception as e:
            print("Error:", e)
            client_socket.sendall("error".encode())
            break
            
    client_socket.close()


if __name__ == '__main__':
    # HOST = Config.serv_addr
    # PORT = Config.serv_port

    # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # server_socket.bind((HOST, PORT))
    # server_socket.listen()

    # print('server start')

    # while True:
    #     print('wait')
    #     client_socket, addr = server_socket.accept()
    #     start_new_thread(threaded, (client_socket, addr))

    localizer('/home/keunmo/workspace/Hierarchical-Localization/datasets/fastfive/query/20221004_153108.jpg')