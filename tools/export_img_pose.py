# import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np

"""
Export Image frame poses from COLMAP SfM output(images.txt)

input format: 
  IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME({idx:04d}.jpg) // world to camera
  POINTS2D[] as (X, Y, POINT3D_ID)
output format: 
  IMAGE_ID(NAME) rx ry rz tx ty tz timestamp(NAME)  # rx ry rz is rotation vector, camera to world
"""
# image ID 0000 4자리 zero filled
# sort by image ID
def pose_extractor(input_path: Path, output_path: Path) -> int:
    count = 0
    with open(input_path, 'r') as f:
        lines = f.readlines()
        with open(output_path, 'w') as f:
            for line in lines:
                line = line.strip()
                if line.startswith('#'):
                    continue
                line = line.split(' ')
                if len(line) > 10: # skip points2d[]
                    continue
                # image_id = line[0]
                qw = line[1]
                qx = line[2]
                qy = line[3]
                qz = line[4]
                tx = line[5]
                ty = line[6]
                tz = line[7]
                # camera_id = line[8]
                name = line[9].split('.')[0]
                # timestamp = name.split('/')[-1].split('.')[0]
                r = np.array([qx, qy, qz, qw], dtype=np.float32)
                print(f"quat_r: {r}")
                r = R.from_quat(r)
                r = r.as_matrix()
                t = np.array([tx, ty, tz], dtype=np.float32)

                print(f"r: {r}")
                print(f"t: {t}")

                # change world2cam to cam2world
                r_inv = r.transpose()
                t_inv = -r_inv.dot(t)
                # print(f"r_inv: {r_inv}")
                # print(f"t_inv: {t_inv}")
                # exit(1)
                r_inv = R.from_matrix(r_inv)
                r_inv = r_inv.as_rotvec()
                f.write(f"{name} {r_inv[0]} {r_inv[1]} {r_inv[2]} {t_inv[0]} {t_inv[1]} {t_inv[2]} {name}\n")
                count+=1
    print('total {} frame pose converted.'.format(count))
    return count


# sort by timestamp
def pose_sorter(pose: Path, output: Path):
    with open(pose, 'r') as f:
        lines = f.readlines()
        lines.sort(key=lambda x: x.split(' ')[-1])
        with open(output, 'w') as f:
            for line in lines:
                f.write(line)
    print('pose sorted by timestamp.')



if __name__ == '__main__':
    base_dir = Path('/home/keunmo/workspace/Hierarchical-Localization/outputs/box_recon/box_images/sfm-txt')
    input_path = base_dir / 'images.txt'
    output_path = base_dir / 'frame_poses2.txt'
    pose_extractor(input_path, output_path)
    sorted_path = base_dir / 'frame_poses_sorted.txt'
    # pose_sorter(output_path, sorted_path)