# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# utils.open3d.py
#
# Author: Changhee Won (changhee.won@multipleye.co)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import *

import open3d as o3d
import open3d.visualization as o3d_vis

from utils.array import *
from utils.image import *
from utils.geometry import *

def make_vis_camera_points(
    cam_size: float, fov_deg: float = 120.0, aspect: float = 0.75) \
        -> np.ndarray:
    cam = np.zeros((3, 16))
    half_fov = np.deg2rad(fov_deg) / 2
    x = cam_size * np.sin(half_fov)
    y = x * aspect
    z = cam_size * np.cos(half_fov)
    cam[0, [1, 7, 8, 13, 14, 15]] = x
    cam[0, [3, 5, 9, 10, 11, 12]] = -x
    cam[1, [1, 3, 8, 9, 10, 15]] = y
    cam[1, [5, 7, 11, 12, 13, 14]] = -y
    cam[2, :] = z
    cam[2, [0, 2, 4, 6]] = 0
    return cam

def make_o3d_line_strip_obj(
        pts: ArrayType, colors: Optional[ArrayType] = None) -> \
            'o3d.geometry.Geometry':
    pts = to_numpy(pts)
    if pts.shape[0] == 3 and pts.shape[1] != 3:
        pts = np.ascontiguousarray(pts.T) # transpose
    pts = pts.reshape((-1, 3)).astype(float)
    npts = pts.shape[0]
    line_strip_ind = [(i, i + 1) for i in range(npts - 1)] 
    obj = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(line_strip_ind))
    if colors is not None: 
        colors = to_numpy(colors)
        if len(colors.shape) == 2 and \
                colors.shape[0] == 3 and colors.shape[1] != 3:
            colors = colors.T
        colors = colors.reshape((-1, 3))
        if colors.dtype == np.uint8:
            colors = colors.astype(float) / 255.0
        if colors.shape[0] == pts.shape[0]: 
            obj.colors = o3d.utility.Vector3dVector(colors)
        elif colors.shape[0] == 1: 
            obj.paint_uniform_color(colors.squeeze()) 
    return obj 

def make_o3d_line_strip_obj_quadcam(
        pts: ArrayType, npts_cam:int, ncams:int) -> \
            'o3d.geometry.Geometry':
    pts = to_numpy(pts) 
    if pts.shape[0] == 3 and pts.shape[1] != 3:
        pts = np.ascontiguousarray(pts.T) # transpose
    pts = pts.reshape((-1, 3)).astype(float)
    # npts = pts.shape[0]
    assert pts.shape[0]==npts_cam*ncams 
    
    line_strip_ind = []
    for j in range(ncams):
        line_strip_ind += [(i+j*npts_cam, i+j*npts_cam + 1) for i in range(npts_cam - 1)] 

    obj = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(line_strip_ind))
    # if colors is not None: 

    assert ncams==4
    colors = [[1.0, 0, 0] for _ in range(npts_cam-1)]
    colors += [[0, 1.0, 0] for _ in range(npts_cam-1)]
    colors += [[0, 0, 1.0] for _ in range(npts_cam-1)]
    colors += [[1.0, 1.0, 0] for _ in range(npts_cam-1)]
    colors = to_numpy(colors)
    if len(colors.shape) == 2 and \
            colors.shape[0] == 3 and colors.shape[1] != 3:
        colors = colors.T
    colors = colors.reshape((-1, 3))
    if colors.dtype == np.uint8:
        colors = colors.astype(float) / 255.0
    
    if colors.shape[0] == pts.shape[0]-ncams: 
        obj.colors = o3d.utility.Vector3dVector(colors)
    elif colors.shape[0] == 1: 
        obj.paint_uniform_color(colors.squeeze())
    return obj 

def make_o3d_pointcloud_obj(
    pts: ArrayType, colors: Optional[ArrayType] = None) -> \
        'o3d.geometry.Geometry':
    pts = to_numpy(pts)
    if pts.shape[0] == 3 and pts.shape[1] != 3:
        pts = np.ascontiguousarray(pts.T) # transpose
    pts = pts.reshape((-1, 3)).astype(float)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        colors = to_numpy(colors)
        if len(colors.shape) == 2 and \
                colors.shape[0] == 3 and colors.shape[1] != 3:
            colors = colors.T
        colors = colors.reshape((-1, 3))
        if colors.dtype == np.uint8:
            colors = colors.astype(float) / 255.0
        if colors.shape[0] == pts.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif colors.shape[0] == 1:
            pcd.paint_uniform_color(colors.squeeze())
    return pcd

def make_o3d_triangle_mesh_obj(vertices: ArrayType, triangles: ArrayType) -> \
        'o3d.geometry.Geometry':
    vertices = to_numpy(vertices)
    triangles = to_numpy(triangles)
    if vertices.shape[0] == 3 and vertices.shape[1] != 3:
        vertices = np.ascontiguousarray(vertices.T) # transpose
    if triangles.shape[0] == 3 and triangles.shape[1] != 3:
        triangles = np.ascontiguousarray(triangles.T) # transpose
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh

def make_o3d_axis_mesh_obj(cam2world: np.ndarray, size:float = 0.3, axis_size:float = 1.0)\
        -> 'o3d.geometry.Geometry':
    transform = to_mat4(cam2world)
    ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if axis_size != 1:
        ax.scale(axis_size,center=ax.get_center())
    ax.transform(transform) 
    return ax

# pcd = o3d.geometry.PointCloud()
# c = color_map('jet', ray[1, :]).astype(float) / 255.0
# pcd.points = o3d.utility.Vector3dVector(ray.T)
# pcd.colors = o3d.utility.Vector3dVector(c.reshape((-1, 3)))
# # pcd.paint_uniform_color([0.5, 0.0, 0.2])
# o3d.visualization.draw([pcd])
# ply_point_cloud = o3d.data.PLYPointCloud()
# pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
# o3d.visualization.draw_geometries([pcd],
#                                 zoom=0.3412,
#                                 front=[0.4257, -0.2125, -0.8795],
#                                 lookat=[2.6172, 2.0475, 1.532],
#                                 up=[-0.0694, -0.9768, 0.2024])
