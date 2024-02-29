import os
import cv2
import numpy as np
from typing import Union, List, Optional
from pathlib import Path
from tqdm import tqdm
import yaml

'''Undistort images using opencv camera model.
data: Path to the dataset
output: Path to the output folder
params: Camera parameters[fx, fy, cx, cy, k1, k2, p1, p2]]
return: Number of undistorted images'''
def undistort_fisheye(data: Path, params: List[float], output: Optional[Path] = None,
                    view: Optional[bool] = False, save: Optional[bool] = True) -> int:
    dataset = os.listdir(data)
    dataset.sort()
    if output is None:
        output = Path(f'{data}_undistorted')
        os.makedirs(output, exist_ok=True)
    elif not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
    D = np.array([params[4], params[5], params[6], params[7]])
    # new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=1.0)
    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    # undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    for i in tqdm(range(len(dataset))):
        file = data / dataset[i]
        img = cv2.imread(str(file))
        undistorted_img = cv2.fisheye.undistortImage(img, K, D, Knew=K)
        # newcamera, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 0)
        # undistorted_img = cv2.undistort(img, K, D, None, newCameraMatrix=K)
        if view:
            cv2.imshow('black', undistorted_img)
            key = cv2.waitKey(0)
            if key == 27:  # press ESC to stop viewing
                view = False
                cv2.destroyAllWindows()
        if save:
            cv2.imwrite(str(output / dataset[i]), undistorted_img)
    
    print("Save {} undistorted images to {}".format(len(os.listdir(output)), output))
    return len(os.listdir(output))


def undistort(data: Path, params: List[float], output: Optional[Path] = None, remove_black = False,
              view: Optional[bool] = False, save: Optional[bool] = True) -> int:
    dataset = os.listdir(data)
    dataset.sort()
    if output is None:
        output = Path(f'{data}_undistorted')
        os.makedirs(output, exist_ok=True)
    elif not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
    D = np.array([params[4], params[5], params[6], params[7]])
    # file = data / dataset[0]
    # img = cv2.imread(str(file))
    # H, W = img.shape[:2]
    for i in tqdm(range(len(dataset))):
        file = data / dataset[i]
        img = cv2.imread(str(file))
        if remove_black:
            new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(K, D, (img.shape[1], img.shape[0]), 0)
            undistorted_img = cv2.undistort(img, K, D, None, new_cam_mat)
        else:
            undistorted_img = cv2.undistort(img, K, D)
        if view:
            cv2.imshow('black', undistorted_img)
            key = cv2.waitKey(0)
            if key == 27:
                view = False
                cv2.destroyAllWindows()
        if save:
            cv2.imwrite(str(output / dataset[i]), undistorted_img)
    print("Save {} undistorted images to {}".format(len(os.listdir(output)), output))
    if remove_black:
        print("New Camera Matrix:")
        print(new_cam_mat)
        # save new cam mat as txt
        with open(output / "intrinsics.txt", 'w') as f:
            f.write("# PINHOLE fx fy cx cy\n")
            f.write(f"{new_cam_mat[0, 0]} {new_cam_mat[1, 1]} {new_cam_mat[0, 2]} {new_cam_mat[1, 2]}")
    return len(os.listdir(output))


def get_intrinsic(calib_path: Path) -> List[float]:
    # read calibration yml file
    with open(calib_path) as f:
        calib = yaml.load(f, Loader=yaml.FullLoader)
        cam_mat = calib["camera_matrix"]["data"] # fx 0 cx 0 fy cy 0 0 1
        dist_coeff = calib["distortion_coefficients"]["data"] # p1 p2 k1 k2 k3
    params = [cam_mat[0], cam_mat[4], cam_mat[2], cam_mat[5]] + dist_coeff[2:4] + dist_coeff[:2] # fx, fy, cx, cy, k1, k2, p1, p2
    # print(params)
    return params



if __name__ == '__main__':
    # data = Path('sampled_5')
    # # output_path = Path('test')
    # # w = 1280
    # # h = 960
    # # camera model = opencv fisheye
    # # camara parameters
    fx = 394.5731183
    fy = 395.4102782
    cx = 496.6285659
    cy = 371.5605722
    k1 = -0.0001751
    k2 = -0.0004413
    p1 = 0.0135703
    p2 = -0.0095865
    params = [fx, fy, cx, cy, k1, k2, p1, p2]
    # undistort_fisheye(dataset_path, params)
    orig_data = Path("/home/keunmo/workspace/Hierarchical-Localization/datasets/hangwon_park_wide/mapping")
    # intrinsic = Path("/home/keunmo/workspace/dataset/s20fe-robjet-uw-43/calibration.yml")
    output_path = Path("/home/keunmo/workspace/Hierarchical-Localization/datasets/hangwon_park_wide/mapping_undistorted")
    # params = get_intrinsic(intrinsic)
    undistort(data=orig_data, params=params, output=output_path, remove_black=True, view=False, save=False)
