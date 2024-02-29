from operator import mod
import os
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from pixsfm.refine_hloc import PixSfM

# from hloc.utils import viz_3d
import pycolmap


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# target = 'hanwon_park_wide_undistorted_spp_sg'
target = "MipNeRF360-bicycle-test"
# images = Path('../datasets/' + target)
# images = Path('../datasets/' + 'hanwon_park_wide_undistorted')
images = Path("/home/keunmo/workspace/Hierarchical-Localization/dataset2/MipNeRF360/bicycle")
outputs = Path('../outputs/' + target)

sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

feature_conf = extract_features.confs['superpoint_aachen']
# feature_conf = extract_features.confs['disk']
# feature_conf = extract_features.confs['sift']
matcher_conf = match_features.confs['superglue']
# matcher_conf = match_features.confs['superpoint+lightglue']
# matcher_conf = match_features.confs['disk+lightglue']
# matcher_conf = match_features.confs['NN-ratio']

references = [p.relative_to(images).as_posix() for p in (images / 'images/').iterdir()]

extract_features.main(feature_conf, images, image_list=references, feature_path=features)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

# intrinsic_path = images / 'intrinsics.txt'
# with open(intrinsic_path, 'r') as f:
#     lines = f.readlines()
#     intrinsics = lines[-1].strip().split(' ')
# intrinsics = str.join(', ', intrinsics)  # fx, fy, cx, cy
intrinsics = "4649.505978, 4627.300373, 2473.000000, 1643.000000"

mapper_options = pycolmap.IncrementalMapperOptions()
mapper_options.ba_refine_focal_length = False
mapper_options.ba_refine_principal_point = False
mapper_options.ba_refine_extra_params = False
mapper_options = mapper_options.todict()

image_options = pycolmap.ImageReaderOptions()
image_options.camera_model = 'PINHOLE'
image_options.camera_params = intrinsics
image_options.todict()

# model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, camera_mode=pycolmap.CameraMode.SINGLE, image_list=references, image_options=image_options, mapper_options=mapper_options)

refiner_conf = {
    "BA": {
        "optimizer": {
            "refine_focal_length": False,
            "refine_principal_point": False,
            "refine_extra_params": False,  # distortion parameters
            "refine_extrinsics": True  # camera poses
        }
    },
    "dense_features": {"use_cache": True}
}

refiner = PixSfM(conf=refiner_conf)
model, debug_outputs = refiner.reconstruction(sfm_dir, images, sfm_pairs, features, matches, camera_mode=pycolmap.CameraMode.SINGLE, image_list=references, image_options=image_options, mapper_options=mapper_options)
