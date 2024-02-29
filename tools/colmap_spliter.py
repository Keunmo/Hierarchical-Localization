import numpy as np
import plyfile
from scipy.spatial import KDTree
from tqdm import tqdm
from pathlib import Path

from hloc.utils.read_write_model import read_model, write_model


def split_sfm(orig_sfm_path: Path, partial_ply: Path, partial_sfm_path=None, thresh=1e-6):
    partial_pcd = plyfile.PlyData.read(partial_ply)
    partial_points = np.vstack([partial_pcd['vertex']['x'],
                                partial_pcd['vertex']['y'],
                                partial_pcd['vertex']['z']]).T
    cameras, images, points3D = read_model(orig_sfm_path.as_posix())
    points3D = list(points3D.values())
    partial_cameras = {}
    partial_images = {}
    partial_points3D = {}
    full_points = np.array([point.xyz for point in points3D])
    tree = KDTree(full_points)
    for point in tqdm(partial_points):
        dist, ind = tree.query(point, k=1)
        if dist > thresh:
            continue
        partial_points3D[points3D[ind].id] = points3D[ind]
        for image_id in points3D[ind].image_ids:
            if image_id not in partial_images:
                partial_images[image_id] = images[image_id]
                if images[image_id].camera_id not in partial_cameras:
                    partial_cameras[images[image_id].camera_id] = cameras[images[image_id].camera_id]
    
    if len(partial_points3D) != len(partial_points):
        print(f'Get partial points: {len(partial_points3D)} out of {len(partial_points)}')
    else:
        print(f'Get all {len(partial_points3D)} partial points')
    
    if partial_sfm_path is None:
        partial_sfm_path = (orig_sfm_path.parent / f'{orig_sfm_path.stem}_partial').as_posix()
    
    Path(partial_sfm_path).mkdir(parents=True, exist_ok=True)
    write_model(partial_cameras, partial_images, partial_points3D, partial_sfm_path)
    print(f'Write partial SfM to {partial_sfm_path}')