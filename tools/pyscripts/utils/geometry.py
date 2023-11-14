# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# utils.geometry.py
#
# Author: Changhee Won (chwon@hanyang.ac.kr)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import *

from utils.common import *
from utils.array import *
from scipy.spatial.transform import Rotation as R

def to_mat4(transform:np.ndarray) -> np.ndarray:
    mat = to_pose_matrix(transform) # 3 x 4
    row = np.array([0, 0, 0, 1], dtype=mat.dtype).reshape((1, 4))
    return np.concatenate((mat, row), 0)

def rodrigues(r: np.ndarray) -> np.ndarray:
    '''
        if mat, convert to vec.
        if vec, convert to mat.
    '''
    if r.size == 3: return R.from_rotvec(r.squeeze()).as_matrix()
    elif r.size == 4: return R.from_quat(r.squeeze()).as_matrix()
    else: return R.from_matrix(r).as_rotvec().reshape((3, 1))

def rodrigues_many(r: np.ndarray) -> np.ndarray:
    '''
        if mat, convert to vec.
        if vec, convert to mat.
    '''
    if r.size == 3: return R.from_rotvec(r.squeeze()).as_matrix()
    elif r.size == 4: return R.from_quat(r.squeeze()).as_matrix()
    else: return R.from_matrix(r).as_rotvec().reshape((3, -1))
    
def to_quat(r: np.ndarray) -> np.ndarray:
    if r.size == 3: return R.from_rotvec(r.squeeze()).as_quat().reshape((4, 1))
    else: return R.from_matrix(r).as_quat().reshape((4, 1))

def get_rot(transform: np.ndarray) -> np.ndarray:
    '''
        args
        transform: camera pose vec or mat
        
        return
        rotational matrix
    '''
    if transform.size == 6:
        transform = transform.reshape((6, 1))
        return rodrigues(transform[:3])
    elif transform.shape == (3, 4) or transform.shape == (4, 4): #rotation matrix
        return transform[:3, :3]
    elif len(transform.shape) == 3: #rotation matrix (w/ batch)
        return transform[:, :3, :3]
    else:
        LOG_ERROR(
            'Invalid shape of input transform: {}'.format(transform.shape))
        return None

def get_tr(transform: np.ndarray) -> np.ndarray:
    if transform.size == 6:
        transform = transform.reshape((6, 1))
        return transform[3:6].reshape((3, 1))
    elif transform.shape == (3, 4) or transform.shape == (4, 4):
        return transform[:3, 3].reshape((3, 1))
    elif len(transform.shape) == 3:
        return transform[:, :3, 3].T.reshape((3, -1))
    else:
        LOG_ERROR(
            'Invalid shape of input transform: {}'.format(transform.shape))
        return None

def to_pose_vector(transform: np.ndarray) -> np.ndarray:
    if transform.size == 6: return transform.reshape((6, 1))
    else:
        return np.concatenate(
            (rodrigues(get_rot(transform)), get_tr(transform)), 0)

def to_pose_vector_many(transform: np.ndarray) -> np.ndarray:
    if transform.size == 6: return transform.reshape((6, 1))
    else:
        return np.concatenate(
            (rodrigues_many(get_rot(transform)), get_tr(transform)), 0)
        
def to_pose_matrix(transform: np.ndarray) -> np.ndarray:
    '''
        Return
        (3,4) pose matrix
    '''
    if transform.size == 6:
        return np.concatenate(
            (get_rot(transform), get_tr(transform)), 1) #get_rot and get_tr return matrix. (3,3) and (3,1)
    else:
        return transform[:3, :4]

def inverse_transform(transform: np.ndarray) -> np.ndarray:
    R, tr = get_rot(transform), get_tr(transform)
    R_inv = R.transpose()
    tr_inv = -R_inv.dot(tr)
    if transform.size == 6:
        r_inv = rodrigues(R_inv)
        return np.concatenate((r_inv, tr_inv), axis=0) # (6, 1) vector
    else:
        return np.concatenate((R_inv, tr_inv), axis=1) # (3, 4) matrix

def merged_transform(t2: np.ndarray, t1: np.ndarray) -> np.ndarray: # T2 * T1
    R1, tr1 = get_rot(t1), get_tr(t1)
    R2, tr2 = get_rot(t2), get_tr(t2)
    R = np.matmul(R2, R1)
    tr = R2.dot(tr1) + tr2
    if t1.size == 6:
        rot = rodrigues(R) #mat to vec
        return np.concatenate((rot, tr), axis=0)
    else:
        return np.concatenate((R, tr), axis=1)

def apply_transform(transform: np.ndarray, P: 'torch.Tensor | np.ndarray') \
        -> 'torch.Tensor | np.ndarray':
    R, tr = get_rot(transform), get_tr(transform)
    if is_torch_tensor(P):
        R = torch.Tensor(R).to(P.device)
        tr = torch.Tensor(tr).to(P.device)
        return torch.matmul(R, P) + tr
    else:
        return R.dot(P) + tr

def compute_pose_translation(
        poses0: np.ndarray, poses1: np.ndarray) -> np.ndarray:
    trs0 = poses0[3:, :]
    trs1 = poses1[3:, :]
    diff = sqrt(sum((trs0 - trs1)**2, 0))
    return diff

def fit_plane(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
        fit a plane with points
        returns the normal vector of the plane and the distance between world origin and the plane
    '''
    if pts.shape[0] != 3 and pts.shape[1] == 3:
        pts = np.ascontiguousarray(pts.T)
    pts = pts.reshape((3, -1))
    center = pts.mean(1).reshape((3, 1)) # avg of point coords

    shifted_pos = (pts - center) # recompute the point coords by the center
    _, _, VH = np.linalg.svd(shifted_pos.T) 
    V = VH.T.conj() # complex conjugate
    n = V[:, -1] # svd's colvec[:,-1] should be the normal vector
    if n[1] < 0: n *= -1 # make up-vector positive. up-vector must be positive
    n = n / np.linalg.norm(n)
    d = sum(n.flatten() * center.flatten()) # ax+by+cz+d=0
    return n.flatten(), -d  

def rotation_from_two_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if v.astype(bool).any(): #if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions

def interpolate_pose(
        pose_prev: np.ndarray, pose_cur: np.ndarray, nframes: int) \
            -> np.ndarray:
    ts = np.linspace(0, 1.0, nframes + 1)
    ts = ts[:-1]
    tr_cur, tr_prev = get_tr(pose_cur), get_tr(pose_prev)
    trs = tr_prev * (1 - ts) + ts * tr_cur

    q1, q0 = to_quat(get_rot(pose_cur)), to_quat(get_rot(pose_prev))
    dot = q1.T.dot(q0)
    if dot < 0:
        dot *= -1.0
        q1 *= -1.0
    theta = acos(dot)
    if theta < 1e-5:
        new_q = q0 * (1 - ts) + ts * q1
    else:
        sin_th = sin(theta)
        new_q = sin((1 - ts) * theta) * q0 + sin(ts * theta) * q1
        new_q /= sin_th
    out_poses = np.zeros((6, nframes))
    for i in range(nframes):
        out_poses[:3, i] = rodrigues(rodrigues(new_q[:, i])).flatten()
        out_poses[3:, i] = trs[:, i]
    return out_poses

def compute_normal_from_depthmap(
        depth: ArrayType, ray: ArrayType) -> ArrayType:
    #ray: direction in world coordinate. (ray and cam2world already considered)
    def compute_normal(p0, p1, p2):
        u, v = p1 - p0, p2 - p0
        n = normalize(cross(u, v, 0))
        return n
    is_torch = is_torch_tensor(depth)
    h, w = depth.shape[0], depth.shape[1]
    #transform to world pt
    ray = ray.reshape((3, h, w))
    if is_torch: ray = to_torch_as(ray, depth)
    pt = depth.reshape((1, h, w)) * ray
    pt_left = roll(pt, 1, 2)
    pt_right = roll(pt, -1, 2)
    pt_up = roll(pt, 1, 1)
    pt_down = roll(pt, -1, 1)
    hmaxs = (1.0 / EPS_SINGLE) * (
        torch.ones((1, w), dtype=depth.dtype, device=depth.device) if is_torch \
            else np.ones((1, w)))
    d_left = roll(depth, 1, axis=1)
    d_right = roll(depth, -1, axis=1)

    diff_left = abs(depth - d_left)
    diff_right = abs(depth - d_right)
    ud_diff = abs((depth[:-1, :] - depth[1:, :]))
    diff_up = concat((hmaxs, ud_diff), 0)
    diff_down = concat((ud_diff, hmaxs), 0)

    is_left = diff_left < diff_right
    is_right = logical_not(is_left)
    is_up = diff_up < diff_down
    is_down = logical_not(is_up)

    normal = zeros_like(pt)
    lu = logical_and(is_left, is_up)
    ru = logical_and(is_right, is_up)
    ld = logical_and(is_left, is_down)
    rd = logical_and(is_right, is_down)

    normal[:, lu] = compute_normal(pt[:, lu], pt_up[:, lu], pt_left[:, lu])
    normal[:, ru] = compute_normal(pt[:, ru], pt_right[:, ru], pt_up[:, ru])
    normal[:, rd] = compute_normal(pt[:, rd], pt_down[:, rd], pt_right[:, rd])
    normal[:, ld] = compute_normal(pt[:, ld], pt_left[:, ld], pt_down[:, ld])
    return normal

def triangulate_rays(
        cam2world_pose0: np.ndarray, cam2world_pose1: np.ndarray,
        cam_rays0: np.ndarray, cam_rays1: np.ndarray) -> np.ndarray:
    us = normalize(get_rot(cam2world_pose0) @ cam_rays0)
    vs = normalize(get_rot(cam2world_pose1) @ cam_rays1)
    tr0 = get_tr(cam2world_pose0)
    tr1 = get_tr(cam2world_pose1)
    t = tr0 - tr1
    uv = (us * vs).sum(0)
    uv_res = 1 - uv * uv

    pts = np.zeros((3, cam_rays0.shape[1]), dtype=float)
    too_small_parallax = uv_res < 1e-9
    ut = (us * t).sum(0)
    vt = (vs * t).sum(0)
    sc = (uv * vt - ut) / uv_res
    tc = (vt - uv * ut) / uv_res
    pts = ((tr0 + sc * us) + (tr1 + tc * vs)) / 2.0
    pts[:, too_small_parallax] = np.nan
    return pts
