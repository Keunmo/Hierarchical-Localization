from operator import mod
import os
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive, match_dense, triangulation
from hloc.visualization import plot_images, read_image
# from hloc.utils import viz_3d
from pixsfm.refine_hloc import PixSfM
import pycolmap


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

target = 'MipNeRF360/bicycle/images'

# images = Path('../data/' + target)
# outputs = Path('../outputs/' + 'MipNeRF360/bicycle/231107_1421')

images = Path("/home/keunmo/workspace/Replica-Dataset/output/apartment_1_cam_arr1")
ref_path = Path("/home/keunmo/workspace/Replica-Dataset/output/apartment_1_cam_arr1/sparse/hloc_sfm_sift_nn/sfm")
"/home/keunmo/workspace/Hierarchical-Localization/datasets/replica_capture/apartment_1_cam_arr1/sparse/hloc_sfm_sift_nn/sfm"

outputs = ref_path / "../../hloc_sfm_loftr"
if not outputs.exists():
    outputs.mkdir(parents=True)

sfm_pairs = outputs / 'pairs-sfm.txt'
# loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
# features_sparse = outputs / 'features_sparse.h5'
# matches = outputs / 'matches.h5'

matcher_conf = match_dense.confs['loftr']

references = [p.relative_to(images).as_posix() for p in (images / 'images').iterdir()]

pairs_from_exhaustive.main(sfm_pairs, image_list=references)
features, matches = match_dense.main(matcher_conf, sfm_pairs, images, outputs, max_kps=8192, overwrite=False)

mapper_options = pycolmap.IncrementalMapperOptions()
mapper_options.ba_refine_focal_length = False
mapper_options.ba_refine_principal_point = False
mapper_options.ba_refine_extra_params = False
mapper_options = mapper_options.todict()

# triangulation.main(sfm_dir, ref_path, images, sfm_pairs, features, matches, mapper_options=mapper_options)

refiner_conf = {
    "BA": {
        "optimizer": {
            "refine_focal_length": False,
            "refine_principal_point": False,
            "refine_extra_params": False,  # distortion parameters
            "refine_extrinsics": False  # camera poses
        }
    },
    "dense_features": {"use_cache": True}
}

refiner = PixSfM(conf=refiner_conf)
# model, debug_outputs = refiner.reconstruction(sfm_dir, images_path, sfm_pairs, features, matches, camera_mode=pycolmap.CameraMode.SINGLE, image_list=references, image_options=image_options, mapper_options=mapper_options)
model, debug_outputs = refiner.triangulation(sfm_dir, ref_path, images, sfm_pairs, features, matches, mapper_options=mapper_options)
