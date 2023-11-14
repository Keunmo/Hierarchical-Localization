# utils.ply.py
#
# Author: Changhee Won (changhee.won@multipleye.co
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import *
from struct import unpack, pack

from utils.geometry import *

class PlyObject:

    def __init__(self, pts:np.ndarray = None, colors:np.ndarray = None):
        self.vertices = []
        self.colors = []
        self.indices = []
        self.normals = []
        self.alphas = []
        self.num_face = 3
        if pts is not None:
            self.setVertices(pts)
        if colors is not None:
            self.setColors(colors)

    def hasVertices(self) -> bool: return len(self.vertices) > 0
    def hasTriangles(self) -> bool: return len(self.indices) > 0
    def hasColors(self) -> bool: return len(self.colors) > 0
    def hasNormals(self) -> bool: return len(self.normals) > 0
    def hasAlpha(self) -> bool: return len(self.alphas) > 0
    def size(self) -> int: return len(self.vertices)

    # arr: 3 * n shape arrays
    def setVertices(self, arr: np.ndarray) -> None:
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        self.vertices = list(arr.T)

    # arr: 3 * n shape arrays
    def setColors(self, arr: np.ndarray) -> None:
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.shape[0] == 4:
            self.colors = list(arr[:3, :].T)
            self.alphas = list(arr[3, :])
        else:
            self.colors = list(arr.T)

    def writePlyFile(self, path: str, comment:str ='', save_as_double=False):
        num_vertices = len(self.vertices)
        f = open(path, 'w')
        f.write('ply\n')
        f.write('format binary_little_endian 1.0\n')
        if len(comment) > 0:
            f.write('comment %s\n' % comment)
        f.write('element vertex %d\n' % num_vertices)
        precision = 'double' if save_as_double else 'float'
        f.write('property %s x\n' % precision)
        f.write('property %s y\n' % precision)
        f.write('property %s z\n' % precision)
        if self.hasNormals():
            f.write('property %s normal_x\n' % precision)
            f.write('property %s normal_y\n' % precision)
            f.write('property %s normal_z\n' % precision)
        if self.hasColors():
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            if self.hasAlpha():
                f.write('property uchar alpha\n')
        if self.hasTriangles():
            num_face = len(self.indices) // self.num_face
            f.write('element face %d\n' % num_face)
            f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        for i in range(self.size()):
            if save_as_double:
                v = self.vertices[i].astype(np.double)
                v.tofile(f)
            else:
                self.vertices[i].tofile(f)
            if self.hasNormals():
                if save_as_double:
                    n = self.normals[i].astype(np.double)
                    n.tofile(f)
                else:
                    self.normals[i].tofile(f)
            if self.hasColors():
                self.colors[i].tofile(f)
                if self.hasAlpha():
                    self.alphas[i].tofile(f)
        if self.hasTriangles() and self.num_face == 3:
            for i in range(0, len(self.indices), 3):
                f.write(pack('B', 3))
                f.write(pack('i', int(self.indices[i])))
                f.write(pack('i', int(self.indices[i+1])))
                f.write(pack('i', int(self.indices[i+2])))
        f.close()



