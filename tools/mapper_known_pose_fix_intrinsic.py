from operator import mod
import os
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive, match_dense, triangulation
# from hloc.visualization import plot_images, read_image
from pixsfm.refine_hloc import PixSfM

# from hloc.utils import viz_3d
import pycolmap


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# target = 'hanwon_park_wide_undistorted_spp_sg'
# target = "MipNeRF360-bicycle-test"
# images_path = Path('../datasets/' + target)
# images_path = Path('../datasets/' + 'hanwon_park_wide_undistorted')

# images_path = Path("/home/keunmo/workspace/Replica-Dataset/output/apartment_1_cam_arr1/sparse/model")
# ref_path = Path("/home/keunmo/workspace/Replica-Dataset/output/apartment_1_cam_arr1/sparse/model")

# output_path = ref_path / "../hloc_sfm_spp_sg"


def sfm_refine_known_cam_poses(ref_path: Path, images_path: Path, output_path: Path):
    sfm_pairs = output_path / 'pairs-sfm.txt'
    loc_pairs = output_path / 'pairs-loc.txt'
    sfm_dir = output_path / 'sfm'
    features = output_path / 'features.h5'
    matches = output_path / 'matches.h5'

    """Use feature point + feature matcher"""
    # feature_conf = extract_features.confs['superpoint_aachen']
    feature_conf = extract_features.confs['superpoint_inloc']
    # feature_conf = extract_features.confs['disk']
    # feature_conf = extract_features.confs['sift']
    matcher_conf = match_features.confs['superglue']
    # matcher_conf = match_features.confs['superpoint+lightglue']
    # matcher_conf = match_features.confs['disk+lightglue']
    # matcher_conf = match_features.confs['NN-ratio']

    references = [p.relative_to(images_path).as_posix() for p in (images_path / 'images').iterdir()]

    extract_features.main(feature_conf, images_path, image_list=references, feature_path=features)
    # extract_features.main(feature_conf, images_path, feature_path=features)
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    # pairs_from_exhaustive.main(sfm_pairs, features=features)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    # intrinsic_path = images_path / 'intrinsics.txt'
    # with open(intrinsic_path, 'r') as f:
    #     lines = f.readlines()
    #     intrinsics = lines[-1].strip().split(' ')
    # intrinsics = str.join(', ', intrinsics)  # fx, fy, cx, cy
    intrinsics = "640 640 640 480"

    mapper_options = pycolmap.IncrementalMapperOptions()
    mapper_options.ba_refine_focal_length = False
    mapper_options.ba_refine_principal_point = False
    mapper_options.ba_refine_extra_params = False
    mapper_options = mapper_options.todict()

    image_options = pycolmap.ImageReaderOptions()
    image_options.camera_model = 'PINHOLE'
    image_options.camera_params = intrinsics
    image_options.todict()

    # model = reconstruction.main(sfm_dir, images_path, sfm_pairs, features, matches, camera_mode=pycolmap.CameraMode.SINGLE, image_list=references, image_options=image_options, mapper_options=mapper_options)
    # triangulation.main(sfm_dir, ref_path, images_path, sfm_pairs, features, matches, mapper_options=mapper_options)

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
    model, debug_outputs = refiner.triangulation(sfm_dir, ref_path, images_path, sfm_pairs, features, matches, mapper_options=mapper_options)


def dense_sfm_refine_known_cam_poses(ref_path: Path, images_path: Path, output_path: Path = None):
    sfm_pairs = output_path / 'pairs-sfm.txt'
    sfm_dir = output_path / 'sfm'
    # features = output_path / 'features.h5'
    # matches = output_path / 'matches.h5'
    if output_path is None:
        output_path = ref_path / "../hloc_sfm_loftr"
    if not output_path.exists():
        output_path.mkdir(parents=True)

    matcher_conf = match_dense.confs['loftr']

    references = [p.relative_to(images_path).as_posix() for p in (images_path / 'images').iterdir()]

    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    features, matches = match_dense.main(matcher_conf, sfm_pairs, images_path, output_path, max_kps=8192, overwrite=False) 


    # intrinsic_path = images_path / 'intrinsics.txt'
    # with open(intrinsic_path, 'r') as f:
    #     lines = f.readlines()
    #     intrinsics = lines[-1].strip().split(' ')
    # intrinsics = str.join(', ', intrinsics)  # fx, fy, cx, cy
    intrinsics = "640 640 640 480"

    image_options = pycolmap.ImageReaderOptions()
    image_options.camera_model = 'PINHOLE'
    image_options.camera_params = intrinsics
    image_options.todict()

    mapper_options = pycolmap.IncrementalMapperOptions()
    mapper_options.ba_refine_focal_length = False
    mapper_options.ba_refine_principal_point = False
    mapper_options.ba_refine_extra_params = False
    mapper_options = mapper_options.todict()

    # model = reconstruction.main(sfm_dir, images_path, sfm_pairs, features, matches, camera_mode=pycolmap.CameraMode.SINGLE, image_list=references, image_options=image_options, mapper_options=mapper_options)
    # triangulation.main(sfm_dir, ref_path, images_path, sfm_pairs, features, matches, mapper_options=mapper_options)

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
    model, debug_outputs = refiner.triangulation(sfm_dir, ref_path, images_path, sfm_pairs, features, matches, mapper_options=mapper_options)


if __name__ == "__main__":
    # dataset_list = ['apartment_1_cam_arr1', 'apartment_1_cam_arr2', 'apartment_1_circle1', 'apartment_1_slam1', 'apartment_1_slam2', 'office_3_cam_arr1', 'office_3_cam_arr2']
    dataset_list = ['room_0_cam_arr1', 'room_0_cam_arr2', 'room_0_circle1', 'room_0_slam1']
    for dataset in dataset_list:
        print(f"== Start triangulate {dataset} ==")
        images_path = Path("/home/keunmo/workspace/Hierarchical-Localization/outputs/replica_capture/" + dataset)
        ref_path = Path("/home/keunmo/workspace/Hierarchical-Localization/outputs/replica_capture/" + dataset + "/sparse/model")
        output_path = ref_path / "../hloc_sfm_loftr"
        dense_sfm_refine_known_cam_poses(ref_path, images_path, output_path)
        # sfm_refine_known_cam_poses(ref_path, images_path, output_path)