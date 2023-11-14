# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# utils.camera.py
#
# Author: Changhee Won (changhee.won@multipleye.co)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import *
from utils.common import *
from utils.array import *
from utils.geometry import *
from utils.image import *

from enum import Enum

import yaml
from scipy import ndimage

class CameraType(Enum):
    PINHOLE = "pinhole"
    OPENCV_FISHEYE = "opencv_fisheye"
    UWFISHEYE = "uwfisheye"
    OCAM = "ocam"
    CYLINDER = "cylinder"
    EQUIRECT = "equirect"
    ORTHO = "ortho"
    OTHER = "other"

class CameraModel:
    def __init__(self):
        self.id = 0
        self.type = CameraType.OTHER
        self.width, self.height = 0, 0
        self.cam2rig = concat(
            (np.identity(3), np.zeros((3, 1))), 1).astype(np.float64)
        self.rig2cam = self.cam2rig.copy()
        self.invalid_mask = None
        self.invalid_mask_file = None
        self.max_theta = -1.0

    def load_from_dict(self, dict: Dict[str, Any]) -> bool:
        try:
            if 'cam_id' in dict.keys():
                self.id = dict['cam_id']
            if self.type == CameraType.OCAM:
                self.height, self.width = dict['image_size']
            else:
                self.width, self.height = dict['image_size']
            if 'pose' in dict.keys():
                self.cam2rig = np.array(dict['pose']).reshape((6, 1))
            elif  'cam2rig_pose' in dict.keys():
                self.cam2rig = np.array(dict['cam2rig_pose']).reshape((6, 1))
            if 'max_theta' in dict.keys():
                self.max_theta = dict['max_theta']
            self.rig2cam = inverse_transform(self.cam2rig)
            return True
        except KeyError as e:
            LOG_ERROR(e)
            return False

    def export_to_dict(self) -> Dict[str, Any]:
        dict = {}
        dict['model'] = self.type.value
        if self.id > 0: dict['cam_id'] = self.id
        dict['image_size'] = [self.height, self.width] \
            if self.type == CameraType.OCAM else [self.width, self.height]
        dict['cam2rig_pose'] = to_pose_vector(self.cam2rig).flatten().tolist()
        return dict

    @property
    def image_size(self):
        return self.height, self.width

    @staticmethod
    def create_from_dict(dict: Dict) -> Union['CameraModel', None]:
        cam_type = CameraType(dict['model'])
        if cam_type is CameraType.OCAM:
            cam = OcamModel()
        elif cam_type is CameraType.UWFISHEYE:
            cam = UWFisheyeModel()
        elif cam_type is CameraType.OPENCV_FISHEYE:
            cam = OpenCVFisheyeModel()
        elif cam_type is CameraType.PINHOLE:
            cam = PinholeModel()
        elif cam_type is CameraType.CYLINDER:
            cam = CylinderCamModel()
        elif cam_type is CameraType.EQUIRECT:
            cam = EquirectCamModel()
        else:
            LOG_ERROR('unknown camera model "%s"' % cam_type)
            return None
        if not cam.load_from_dict(dict):
            return None
        return cam

    def pixel2ray(self, pts2d: ArrayType) -> ArrayType:
        raise NotImplementedError
    def ray2pixel(self, pts3d: ArrayType) -> ArrayType:
        raise NotImplementedError

    def is_in_image(self, px: ArrayType) -> bool:
        if (px < 0).any() or \
            logical_or(
                px[0, :] > self.width - 1, px[1, :] > self.height - 1).any():
            return False
        return True

    def pixel_grid(self) -> np.ndarray:
        xs, ys = np.meshgrid(range(self.width), range(self.height))
        pts2d = concat((xs.reshape((1, -1)), ys.reshape((1, -1))), 0)
        return pts2d

    def make_fov_mask(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        pts2d = self.pixel_grid()
        pts3d = self.pixel2ray(pts2d)
        pt = self.ray2pixel(pts3d)
        valid_mask = logical_or(pt[0,:] == -1, pt[1,:] == -1)
        valid_mask = valid_mask.reshape((self.height, self.width))
        if mask is not None :
            valid_mask = logical_or(valid_mask, mask.squeeze())
        return valid_mask

    def get_radii_thetas(self, pix2ray: bool = True, num_sample:int = 500) \
            -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class PinholeModel(CameraModel):
    def __init__(self):
        super().__init__()
        self.xc, self.yc, self.fx, self.fy = 0, 0, 1, 1
        self.k = np.zeros(0)
        self.max_theta = -1.0
        self.alpha = 0
        self.type = CameraType.PINHOLE
        self.norm_type=''

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, self.alpha, self.xc],
             [0, self.fy, self.yc],
             [0, 0, 1]])

    def load_from_dict(self, dict: Dict[str, Any]) -> bool:
        if not super().load_from_dict(dict): return False
        try:
            self.fx, self.fy = dict['focal_length']
            self.xc, self.yc = dict['center']
            if 'lens_distort' in dict.keys():
                self.k = np.zeros(5)
                for i, d in enumerate(dict['lens_distort']):
                    self.k[i] = d
            if 'alpha' in dict.keys():
                self.alpha = dict['alpha']
            self.max_theta = np.deg2rad(dict['max_fov'] / 2.0)
            if 'invalid_mask' in dict.keys():
                self.invalid_mask_file = dict['invalid_mask']
            else:
                self.invalid_mask_file = None
            return True
        except KeyError as e:
            LOG_ERROR(e)
            return False

    def export_to_dict(self) -> Dict[str, Any]:
        dict = super().export_to_dict()
        dict['focal_length'] = [self.fx, self.fy]
        dict['center'] = [self.xc, self.yc]
        if len(self.k) > 0:
            dict['lens_distort'] = self.k.tolist()
        if self.alpha != 0: dict['alpha'] = self.alpha
        dict['max_fov'] = np.rad2deg(self.max_theta * 2.0)
        return dict

    def __apply_len_distortion(self, xs: ArrayType, ys: ArrayType) \
            -> Tuple[ArrayType, ArrayType]:
        xs2 = xs**2
        ys2 = ys**2
        xy = xs * ys
        r = xs2 + ys2
        rad_dist = r * (self.k[0] + self.k[1] * r)
        xs_out = xs * rad_dist + 2 * self.k[2] * xy + self.k[3] * (r + 2 * xs2)
        ys_out = ys * rad_dist + 2 * self.k[3] * xy + self.k[2] * (r + 2 * ys2)
        return xs_out, ys_out

    def pixel2ray(self, pts2d: ArrayType) -> ArrayType:
        xs = (pts2d[0, :] - self.xc) / self.fx
        ys = (pts2d[1, :] - self.yc) / self.fy
        zs = ones_like(xs)
        if len(self.k) > 0:
            dx, dy = xs, ys
            for i in range(8):
                xs2, ys2 = self.__apply_len_distortion(dx, dy)
                dx, dy = xs - xs2, ys - ys2
            xs, ys = dx, dy
        ray = concat(
            (xs.reshape((1, -1)), ys.reshape((1, -1)), zs.reshape((1, -1))), 0)
        return ray 

    def ray2pixel(self, pts3d: ArrayType) -> ArrayType:
        zs = pts3d[2, :]
        xs = pts3d[0, :] / zs
        ys = pts3d[1, :] / zs
        if len(self.k) > 0:
            dx, dy = self.__apply_len_distortion(xs, ys)
            xs += dx
            ys += dy
        if self.alpha != 0:
            xs = self.fx * (xs + self.alpha * ys) + self.xc
        else:
            xs = self.fx * xs + self.xc
        ys = self.fy * ys + self.yc
        return concat((xs.reshape((1, -1)), ys.reshape((1, -1))), 0)

    @staticmethod
    def create_perspective_camera(
            width: int, height: int, wfov_deg: float) -> 'PinholeModel':
        aspect = width / float(height)
        hfov_deg = wfov_deg / aspect
        model = PinholeModel()
        model.xc, model.yc = (width - 1) / 2.0, (height - 1) / 2.0
        model.fy = model.fx = model.xc / tan(np.deg2rad(wfov_deg / 2.0))
        hfov_deg = np.rad2deg(atan2(model.yc, model.fy)) * 2.0
        model.max_theta = np.deg2rad(max(wfov_deg, hfov_deg) / 2.0)
        model.width, model.height = int(width), int(height)
        return model

class OpenCVFisheyeModel(CameraModel):
    def __init__(self):
        super().__init__()
        self.xc, self.yc, self.fx, self.fy = 0, 0, 1, 1
        self.k = np.zeros(0)
        self.max_theta = -1.0
        self.alpha = 0
        self.type = CameraType.OPENCV_FISHEYE
        self.invpol = None

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, self.alpha, self.xc],
             [0, self.fy, self.yc],
             [0, 0, 1]])

    def load_from_dict(self, dict: Dict[str, Any]) -> bool:
        if not super().load_from_dict(dict): return False
        try:
            self.fx, self.fy = dict['focal_length']
            self.xc, self.yc = dict['center']
            self.k = np.array(dict['distortion'])
            if 'alpha' in dict.keys():
                self.alpha = dict['alpha']
            self.max_theta = np.deg2rad(dict['max_fov'] / 2.0)
            if 'invalid_mask' in dict.keys():
                self.invalid_mask_file = dict['invalid_mask']
            else:
                self.invalid_mask_file = None
            if 'inv_poly' in dict.keys():
                num_invpol = dict['inv_poly'][0]
                if len(dict['inv_poly']) - 1 != num_invpol :
                    LOG_WARNING(
                        'Number of coeffs does not match in ocam\'s inv_poly')
                self.invpol = dict['inv_poly'][-1:0:-1] # make reverse
                self.invpol.append(0)
            return True
        except KeyError as e:
            LOG_ERROR(e)
            return False

    def pixel2ray(self, pts2d: ArrayType) -> ArrayType:
        xs = (pts2d[0, :] - self.xc) / self.fx
        ys = (pts2d[1, :] - self.yc) / self.fy
        zs = ones_like(xs)
        k = self.k
        r = sqrt(xs**2 + ys**2)
        converged = zeros_like(r)
        if self.invpol is not None:
            theta = polyval(self.invpol, r)
            converged += 1
        else:
            r[r < -np.pi/2] = -np.pi/2
            r[r > np.pi/2] = np.pi/2
            if is_torch_tensor(r):
                theta = r.clone()
            else:
                theta = r.copy()
            converged[abs(r) <= 1e-8] = True
            for i in range(20):
                targets = converged == 0
                if targets.sum() == 0:
                    break
                r_ = r[targets]
                th = theta[targets]
                th2 = th**2
                th_fix = \
                    (th * (1 + th2 * (
                        k[0] + th2 * (k[1] + th2 * (
                            k[2] + th2 * k[3])))) - r_) / \
                    (1 + 3 * th2 * (
                        k[0] + 5 * th2 * (k[1] + 7 * th2 * (
                            k[2] + 9 * th2 * k[3]))))
                th = th - th_fix
                theta[targets] = th
                conv = abs(th_fix) <= 1e-8
                if is_torch_tensor(targets):
                    targets[targets.clone()] = conv
                else:
                    targets[targets] = conv
                converged[targets] = True
        scale = tan(theta) / (r + EPS)
        xs = xs * scale
        ys = ys * scale
        out = concat(
            (xs.reshape((1, -1)), ys.reshape((1, -1)), zs.reshape((1, -1))), 0)
        norm = sqrt((out**2).sum(0)).reshape((1, -1))
        out = out / norm
        invalid = logical_or(logical_not(converged),
            logical_and(r < 0, theta > 0), logical_and(r > 0, theta < 0),
            theta > self.max_theta)
        out[:, invalid] = np.nan
        return out

    def ray2pixel(self, pts3d: ArrayType) -> ArrayType:
        zs = pts3d[2, :]
        xs = pts3d[0, :] / zs
        ys = pts3d[1, :] / zs
        k = self.k
        r = sqrt(xs**2 + ys**2)
        theta = atan(r)
        th2 = theta**2
        new_r = theta * (
            1 + th2 * (k[0] + th2 * (k[1] + th2 * (k[2] + th2 * k[3]))))
        new_r = new_r / (r + EPS)
        xs = new_r * xs
        ys = new_r * ys
        if self.alpha != 0:
            xs = self.fx * (xs + self.alpha * ys) + self.xc
        else:
            xs = self.fx * xs + self.xc
        ys = self.fy * ys + self.yc
        invalid = logical_or(zs < 0, theta > self.max_theta)
        xs[invalid] = -1.0
        ys[invalid] = -1.0
        return concat((xs.reshape(1, -1), ys.reshape(1, -1)), 0)

    def get_radii_thetas(self, pix2ray: bool = True, num_sample: int = 500) \
            -> Tuple[np.ndarray, np.ndarray]:
        k = self.k
        if pix2ray:
            max_r = max(self.xc, self.width - 1 - self.xc) / self.fx
            step = max_r / (num_sample - 1)
            radii = np.arange(0, max_r + step, step)
            converged = zeros_like(radii)
            if self.invpol is not None:
                thetas = polyval(self.invpol, radii)
                converged += 1
            else:
                radii[radii < -np.pi/2] = -np.pi/2
                radii[radii > np.pi/2] = np.pi/2
                thetas = radii.copy()
                converged[abs(radii) <= 1e-8] = True
                for _ in range(20):
                    targets = converged == 0
                    if targets.sum() == 0:
                        break
                    r_ = radii[targets]
                    th = thetas[targets]
                    th2 = th**2
                    th_fix = \
                        (th * (1 + th2 * (
                            k[0] + th2 * (
                                k[1] + th2 * (k[2] + th2 * k[3])))) - r_) / \
                        (1 + 3 * th2 * (
                            k[0] + 5 * th2 * (k[1] + 7 * th2 * (
                                k[2] + 9 * th2 * k[3]))))
                    th = th - th_fix
                    thetas[targets] = th
                    conv = abs(th_fix) <= 1e-8
                    if is_torch_tensor(targets):
                        targets[targets.clone()] = conv
                    else:
                        targets[targets] = conv
                    converged[targets] = True
            valid = thetas <= self.max_theta
            thetas = thetas[valid]
            radii = radii[valid]
            return radii * self.fx, thetas
        else:
            step = self.max_theta / (num_sample - 1)
            thetas = np.arange(0, self.max_theta + step, step)
            th2 = thetas**2
            new_r = thetas * (
                1 + th2 * (k[0] + th2 * (k[1] + th2 * (k[2] + th2 * k[3]))))
            radii = new_r * self.fx
            return radii, thetas

    def compute_inverse_polynomial(self,
            min_deg:int = 6, max_deg: int = 21, max_px_err:float = 0.001):
        radii, thetas = self.get_radii_thetas(
            pix2ray=False, num_sample=1000)
        px = self.pixel_grid()
        pt = (px - np.array([[self.xc], [self.yc]])) \
            / np.array([[self.fx], [self.fy]])
        r = sqrt((pt**2).sum(0))
        valid = logical_and(r > -np.pi/2, r < np.pi/2)
        invpol = None
        for n in range(min_deg, max_deg):
            invpol = polyfit(radii / self.fx, thetas, n, zero_coeffs=[0])
            ## this part can replace pixel2ray()
            th = polyval(invpol, r)
            valid_ = logical_and(valid, th <= self.max_theta)
            new_r = tan(th) / (r + EPS)
            ray = hom(pt * new_r.reshape((1, -1)))
            ##
            px2 = self.ray2pixel(ray)
            diff = sqrt(((px - px2)**2).sum(0))
            max_diff = diff[valid_].max()
            if max_diff <= max_px_err:
                break
        LOG_INFO('# For %d-th degree, max error: %.4f px' % (
            len(invpol) - 1, max_diff))
        self.invpol = invpol

    def convert_to_uwfisheye(self,
            max_px_err: float = 0.001) -> 'UWFisheyeModel':
        ucam = UWFisheyeModel()
        ucam.height, ucam.width = self.image_size
        ucam.xc, ucam.yc = self.xc, self.yc
        ucam.max_theta = self.max_theta
        ucam.cam2rig, ucam.rig2cam = self.cam2rig, self.rig2cam
        ucam.f = (self.fx + self.fy) / 2
        if self.invpol is None:
            self.compute_inverse_polynomial(max_px_err=max_px_err)
        px = self.pixel_grid()
        ray_self = self.pixel2ray(px)
        valid = logical_not(isnan(ray_self[0,:]))
        ray_self = ray_self[:, valid]
        px = px[:, valid]
        px2 = self.ray2pixel(ray_self)
        # pol for pix2ray
        k = self.k
        ucam.pol = np.array([k[3], 0, k[2], 0, k[1], 0, k[0], 0, 1, 0])
        px3 = ucam.ray2pixel(ray_self)
        # ucam.pol = polyfit(thetas, new_r, n, [0])
        ucam.invpol = self.invpol
        ray_self = self.pixel2ray(px)
        ray_ucam = ucam.pixel2ray(px)
        ray_diff_theta = 2 * asin(sqrt(((ray_self - ray_ucam)**2).sum(0)) / 2.0)
        px_ucam = ucam.ray2pixel(ray_self)
        px_diff = sqrt(((px - px_ucam)**2).sum(0))
        LOG_INFO(
            '# convert_to_uwfisheye max_px_err: %.2f, max_deg_err: %.2f' % (
                px_diff.max(), np.rad2deg(ray_diff_theta.max())))
        return ucam

class UWFisheyeModel(CameraModel):

    def __init__(self):
        super().__init__()
        self.xc, self.yc, self.f = 0, 0, 1
        self.pol = np.zeros(0)
        self.invpol = np.zeros(0)
        self.max_theta = -1.0
        self.type = CameraType.UWFISHEYE

    def load_from_dict(self, dict: Dict[str, Any]) -> bool:
        '''
            Read capture.yaml and save information as property.
        '''
        if not super().load_from_dict(dict): return False
        try:
            num_pol = dict['poly'][0]
            if len(dict['poly']) - 1 != num_pol :
                LOG_WARNING(
                    'Number of coeffs does not match in ocam\'s poly')
            self.pol = dict['poly'][-1:0:-1] # make reverse
            self.pol.append(0)
            self.pol = np.array(self.pol)
            num_invpol = dict['inv_poly'][0]
            if len(dict['inv_poly']) - 1 != num_invpol :
                LOG_WARNING(
                    'Number of coeffs does not match in ocam\'s inv_poly')
            self.invpol = dict['inv_poly'][-1:0:-1] # make reverse
            self.invpol.append(0)
            self.invpol = np.array(self.invpol)
            self.f, self.xc, self.yc = dict['intrinsic']
            self.max_theta = float(np.deg2rad(dict['max_theta']))
            if 'invalid_mask' in dict.keys():
                self.invalid_mask_file = dict['invalid_mask']
            else:
                self.invalid_mask_file = None
            return True
        except KeyError as e:
            LOG_ERROR(e)
            return False

    def export_to_dict(self) -> Dict[str, Any]:
        dict = super().export_to_dict()
        num_pol = len(self.pol) - 1 # except 0
        dict['poly'] = [num_pol] + self.pol[-2::-1].tolist()
        num_invpol = len(self.invpol) - 1 # except 0
        dict['inv_poly'] = [num_invpol] + self.invpol[-2::-1].tolist()
        dict['intrinsic'] = [self.f, self.xc, self.yc]
        dict['max_theta'] = float(np.rad2deg(self.max_theta))
        if self.invalid_mask_file is not None:
            dict['invalid_mask'] = self.invalid_mask_file
        return dict

    def pixel2ray(self, pts2d: ArrayType) -> ArrayType:
        x = (pts2d[0, :] - self.xc) / self.f #image plane into normalized image plane
        y = (pts2d[1, :] - self.yc) / self.f
        r = sqrt(x**2 + y**2) 
        theta = polyval(self.invpol, r) #(11,), (num_rays, ) -> (num_rays,)
        new_r = sin(theta) / (r + EPS)
        x = new_r * x.reshape((1, -1))
        y = new_r * y.reshape((1, -1))
        z = cos(theta).reshape((1, -1)) 
        rays = concat((x, y, z), 0) 
        is_nan = logical_or(isnan(x), isnan(y)).flatten()
        rays[:, is_nan] = np.nan
        if self.max_theta > 0:
            rays[:, theta > self.max_theta] = np.nan
        return rays

    def ray2pixel(self, pts3d: ArrayType):
        return self.ray2pixel(pts3d, True)

    def ray2pixel(self,
            pts3d: ArrayType, use_invalid_mask: bool = True) -> ArrayType:
        x = pts3d[0, :]
        y = pts3d[1, :]
        z = pts3d[2, :]
        r = sqrt(x**2 + y**2)
        theta = atan2(r, z)
        new_r = polyval(self.pol, theta).reshape((1, -1)) / (r + EPS) * self.f
        px = new_r * x.reshape((1, -1)) + self.xc
        py = new_r * y.reshape((1, -1)) + self.yc
        pix = concat((px, py), 0)
        is_nan = logical_or(isnan(x), isnan(y), isnan(z)).flatten()
        pix[:, is_nan] = -1
        if self.max_theta > 0:
            pix[:, theta > self.max_theta] = -1
        if use_invalid_mask and self.invalid_mask is not None:
            hv, wv = self.invalid_mask.shape
            invalid_mask = self.invalid_mask.flatten()
            px = (pix[0, :] * wv / self.width).round().squeeze()
            py = (pix[1, :] * hv / self.height).round().squeeze()
            if is_torch_tensor(pts3d):
                invalid_mask = torch.tensor(invalid_mask, device=pts3d.device)
                px, py = px.long(), py.long()
            else:
                px, py = px.astype(np.int32), py.astype(np.int32)
            is_in_image = logical_and(
                px >= 0, px < wv, py >= 0, py < hv)
            idxs = px[is_in_image] + py[is_in_image] * wv
            if is_torch_tensor(is_in_image):
                is_in_image[is_in_image.clone()] = invalid_mask[idxs] > 0
            else:
                is_in_image[is_in_image] = invalid_mask[idxs] > 0
            pix[:, is_in_image] = -1.0
        return pix

    def get_radii_thetas(self, pix2ray: bool = True, num_sample: int = 500) \
            -> Tuple[np.ndarray, np.ndarray]:
        if pix2ray:
            max_x = max(self.xc, self.width - 1 - self.xc)
            max_y = max(self.yc, self.height - 1 - self.yc)
            max_r = sqrt(max_x**2 + max_y**2)
            step = max_r / (num_sample - 1)
            radii = np.arange(0, max_r + step, step)
            thetas = polyval(self.invpol, radii / self.f)
            valid = thetas <= self.max_theta
            thetas = thetas[valid]
            radii = radii[valid]
            return radii, thetas
        else:
            step = self.max_theta / (num_sample - 1)
            thetas = np.arange(0, self.max_theta + step, step)
            radii = polyval(self.pol, thetas) * self.f
            return radii, thetas

    def convert_to_ocam(self,
            npol: int = 6, min_ninvpol: int = 12,
            max_px_err: float = 0.001) -> 'OcamModel':
        ocam = OcamModel()
        ocam.width, ocam.height = self.width, self.height
        ocam.xc, ocam.yc = self.yc, self.xc # flip
        ocam.max_theta = self.max_theta
        ocam.cam2rig, ocam.rig2cam = self.cam2rig, self.rig2cam
        max_r = sqrt(
            max(self.width - 1 - self.xc, self.xc,
                self.height - 1 - self.yc, self.yc)**2)
        rs_ocam = np.arange(0, max_r, 0.1)
        rs_self = rs_ocam / self.f
        thetas = polyval(self.invpol, rs_self)
        valid = thetas <= self.max_theta
        zs_self = cos(thetas)
        zs_ocam = -zs_self * (rs_ocam) / (sin(thetas) + EPS)
        ocam.pol = polyfit(rs_ocam[valid], zs_ocam[valid], npol, [1])
        zs_ = polyval(ocam.pol, rs_ocam)
        thetas_ = atan2(rs_ocam, -zs_)
        theta_diff = np.rad2deg(abs(thetas - thetas_))[valid]
        LOG_INFO('# Theta error, avg: %.2f, max: %.2f (deg)' %
            (theta_diff.mean(), theta_diff.max()))
        thetas_ocam = thetas - np.pi / 2
        for n in range(min_ninvpol, 21):
            ocam.inv_pol = polyfit(thetas_ocam[valid], rs_ocam[valid], n)
            rs_ = polyval(ocam.inv_pol, thetas_ocam)
            diff = abs(rs_ocam - rs_)
            max_diff = diff[valid].max()
            if max_diff <= max_px_err:
                LOG_INFO('# For %d-th degree, max error: %.4f px', n, max_diff)
                break
        return ocam

    @staticmethod
    def create_equidistant_camera(width: int, max_fov_deg: float) -> 'UWFisheyeModel':
        model = UWFisheyeModel()
        fov_2 = np.deg2rad(max_fov_deg / 2.0)
        xc = (width - 1) / 2.0
        model.f = model.xc = model.yc = xc
        model.max_theta = fov_2
        model.invpol = np.array([fov_2, 0], dtype=np.float64)
        model.pol = np.array([1.0 / fov_2, 0], dtype=np.float64)
        model.width = model.height = width
        return model

class OcamModel(CameraModel):
    def __init__(self):
        super().__init__()
        self.xc, self.yc = 0, 0
        self.c, self.d, self.e = 1, 0, 0
        self.max_theta = np.pi
        self.pol = np.zeros(0)
        self.inv_pol = np.zeros(0)
        self.type = CameraType.OCAM

    def load_from_dict(self, dict: Dict[str, Any]) -> bool:
        if not super().load_from_dict(dict): return False
        try:
            num_pol = dict['poly'][0]
            if len(dict['poly']) - 1 != num_pol :
                LOG_WARNING(
                    'Number of coeffs does not match in ocam\'s poly')
            self.pol = dict['poly'][-1:0:-1] # make reverse
            num_invpol = dict['inv_poly'][0]
            if len(dict['inv_poly']) - 1 != num_invpol :
                LOG_WARNING(
                    'Number of coeffs does not match in ocam\'s inv_poly')
            self.inv_pol = dict['inv_poly'][-1:0:-1] # make reverse
            self.xc, self.yc = dict['center'] # x, y fliped
            self.c, self.d, self.e = dict['affine']
            self.max_theta = np.deg2rad(dict['max_fov']) / 2.0
            if 'invalid_mask' in dict.keys():
                self.invalid_mask_file = dict['invalid_mask']
            else:
                self.invalid_mask_file = None
            return True
        except KeyError as e:
            LOG_ERROR(e)
            return False

    def pixel2ray(self, pts2d: ArrayType) -> ArrayType:
        return self.pixel2ray(pts2d, False, self.max_theta)
    def ray2pixel(self, pts3d: ArrayType) -> ArrayType:
        return self.ray2pixel(pts3d, False, self.max_theta, True)

    def pixel2ray(self,
            pts2d: ArrayType, out_theta: bool = False,
            max_theta: Optional[float] = None) -> ArrayType:
        if max_theta is None: max_theta = self.max_theta
        # flip axis
        x = pts2d[1,:].reshape((1, -1)) - self.xc
        y = pts2d[0,:].reshape((1, -1)) - self.yc
        p = concat((x, y), axis=0)
        invdet = 1.0 / (self.c - self.d * self.e)
        A_inv = invdet * np.array([
            [      1, -self.d],
            [-self.e,  self.c]])
        p = A_inv.dot(p)
        # flip axis
        x = p[1,:].reshape((1, -1))
        y = p[0,:].reshape((1, -1))
        rho = sqrt(x * x + y * y)
        z = polyval(self.pol, rho).reshape((1, -1))
        # theta is angle from the optical axis.
        theta = atan2(rho, -z)
        out = concat((x, y, -z), axis=0)
        norm = sqrt((out**2).sum(0)).reshape((1, -1))
        out = out / norm
        out[:,theta.squeeze() > max_theta] = np.nan
        if out_theta:
            return out, theta
        else:
            return out
    # end pixel2ray

    def ray2pixel(self,
            pts3d: ArrayType, out_theta: bool = False,
            max_theta: Optional[float] = None,
            use_invalid_mask: bool = True) -> ArrayType:
        if max_theta is None: max_theta = self.max_theta
        norm = sqrt(pts3d[0,:]**2 + pts3d[1,:]**2) + EPS
        theta = atan2(-pts3d[2,:], norm)
        rho = polyval(self.inv_pol, theta)
        # max_theta check : theta is the angle from xy-plane in ocam,
        # thus add pi/2 to compute the angle from the optical axis.
        theta = theta + np.pi / 2
        # flip axis
        x = pts3d[1,:] / norm * rho
        y = pts3d[0,:] / norm * rho
        x2 = x * self.c + y * self.d + self.xc
        y2 = x * self.e + y          + self.yc
        x2 = x2.reshape((1, -1))
        y2 = y2.reshape((1, -1))
        out = concat((y2, x2), axis=0)
        out[:, isnan(pts3d[0,:])] = -1.0
        out[:, theta.squeeze() > max_theta] = -1.0
        if use_invalid_mask and self.invalid_mask is not None:
            hv, wv = self.invalid_mask.shape
            invalid_mask = self.invalid_mask.flatten()
            px = (y2 * wv / self.width).round().squeeze()
            py = (x2 * hv / self.height).round().squeeze()
            if is_torch_tensor(pts3d):
                invalid_mask = torch.tensor(invalid_mask, device=pts3d.device)
                px, py = px.long(), py.long()
            else:
                px, py = px.astype(np.int32), py.astype(np.int32)
            is_in_image = logical_and(
                px >= 0, px < wv, py >= 0, py < hv)
            idxs = px[is_in_image] + py[is_in_image] * wv
            # if type(invalid_mask) == torch.Tensor:
            #     is_in_image[is_in_image] = (invalid_mask[idxs] > 0).clone()
            # else:
            if is_torch_tensor(is_in_image):
                is_in_image[is_in_image.clone()] = invalid_mask[idxs] > 0
            else:
                is_in_image[is_in_image] = invalid_mask[idxs] > 0
            out[:, is_in_image] = -1.0

        if out_theta:
            return out, theta
        else:
            return out
    # end ray2pixel

    def get_radii_thetas(self, pix2ray: bool = True, num_sample: int = 500) \
            -> Tuple[np.ndarray, np.ndarray]:
        if pix2ray:
            max_x = max(self.xc, self.width - 1 - self.xc)
            max_y = max(self.yc, self.height - 1 - self.yc)
            max_r = sqrt(max_x**2 + max_y**2)
            step = max_r / (num_sample - 1)
            radii = np.arange(0, max_r + step, step)
            zs = polyval(self.pol, radii)
            thetas = atan2(radii, -zs)
            valid = thetas <= self.max_theta
            thetas = thetas[valid]
            radii = radii[valid]
            return radii, thetas
        else:
            step = self.max_theta / (num_sample - 1)
            thetas = np.arange(0, self.max_theta + step, step)
            radii = polyval(self.inv_pol, thetas - (np.pi / 2))
            return radii, thetas

    def export_to_dict(self) -> Dict[str, Any]:
        dict = super().export_to_dict()
        num_pol = len(self.pol)
        dict['poly'] = [num_pol] + self.pol[-1::-1].tolist()
        num_invpol = len(self.inv_pol)
        dict['inv_poly'] = [num_invpol] + self.inv_pol[-1::-1].tolist()
        dict['center'] = [self.xc, self.yc]
        dict['affine'] = [self.c, self.d, self.e]
        dict['max_fov'] = float(np.rad2deg(self.max_theta) * 2.0)
        if self.invalid_mask_file is not None:
            dict['invalid_mask'] = self.invalid_mask_file
        return dict

class CylinderCamModel(CameraModel):
    def __init__(self):
        super().__init__()
        self.xc, self.yc, self.fy = 0, 0, 1
        self.max_theta = 0
        self.type = CameraType.CYLINDER

    def load_from_dict(self, dict: Dict[str, Any]) -> bool:
        if not super().load_from_dict(dict): return False
        try:
            self.xc, self.yc = dict['center']
            self.fy = dict['focal_length']
            self.max_theta = np.deg2rad(dict['max_fov'] / 2.0)
            return True
        except KeyError as e:
            LOG_ERROR(e)
            return False

    def pixel2ray(self, pts2d: ArrayType) -> ArrayType:
        thetas = (pts2d[0, :] - self.xc) / self.xc * self.max_theta + np.pi / 2
        ys = (pts2d[1, :] - self.yc) / self.fy
        xs = -cos(thetas)
        zs = sin(thetas)
        ray = concat(
            (xs.reshape((1, -1)), ys.reshape((1, -1)), zs.reshape((1, -1))), 0)
        return ray

    def ray2pixel(self, pts3d: ArrayType) -> ArrayType:
        xs = pts3d[0, :]
        ys = pts3d[1, :]
        zs = pts3d[2, :]
        r = sqrt(xs**2 + zs**2)
        ys = (ys / r) * self.fy + self.yc
        theta = atan2(zs, -xs)
        xs = (theta - np.pi / 2) / self.max_theta * self.xc + self.xc
        return concat((xs.reshape((1, -1)), ys.reshape((1, -1))), 0)

    def export_to_dict(self) -> Dict[str, Any]:
        dict = super().export_to_dict()
        dict['focal_length'] = self.fy
        dict['center'] = [self.xc, self.yc]
        dict['max_fov'] = np.rad2deg(self.max_theta * 2.0)
        return dict

    @staticmethod
    def create_cylinder_camera(
            width: int, height: int, wfov_deg: float, hfov_deg: float = -1.0):
        if hfov_deg <= 0:
            hfov_deg = wfov_deg * height / width
        model = CylinderCamModel()
        model.xc, model.yc = (width - 1) / 2.0, (height - 1) / 2.0
        model.max_theta = np.deg2rad(wfov_deg / 2.0)
        model.fy = model.yc / tan(np.deg2rad(hfov_deg / 2.0))
        model.width, model.height = int(width), int(height)
        return model

class EquirectCamModel(CameraModel):
    def __init__(self):
        super().__init__()
        self.phi_min_deg, self.phi_max_deg = 0.0, 0.0
        self.type = CameraType.EQUIRECT

    def load_from_dict(self, dict: Dict[str, Any]) -> bool:
        if not super().load_from_dict(dict): return False
        try:
            self.phi_min_deg, self.phi_max_deg = dict['phi_deg']
            return True
        except KeyError as e:
            LOG_ERROR(e)
            return False

    def export_to_dict(self) -> Dict[str, Any]:
        dict = super().export_to_dict()
        dict['phi_deg'] = [self.phi_min_deg, self.phi_max_deg]
        return dict

    def pixel2ray(self, pts2d: ArrayType) -> ArrayType:
        h, w = self.height, self.width
        xc = w / 2.0
        yc = (h - 1) / 2.0
        thetas = (pts2d[0, :] - xc) / xc * np.pi
        med = np.deg2rad((self.phi_max_deg - self.phi_min_deg) / 2.0)
        med2 = np.deg2rad((self.phi_max_deg + self.phi_min_deg) / 2.0)
        phis = (pts2d[1, :] - yc) / yc * med - med2
        cos_phis = cos(phis)
        xs = sin(thetas) * cos_phis
        zs = cos(thetas) * cos_phis
        ys = sin(phis)
        ray = concat(
            (xs.reshape((1, -1)), ys.reshape((1, -1)), zs.reshape((1, -1))), 0)
        invalid = logical_or(phis < self.phi_min_deg, phis > self.phi_max_deg)
        ray[:, invalid] = np.nan
        return ray

    def ray2pixel(self, pts3d: ArrayType) -> ArrayType:
        h, w = self.height, self.width
        ray = normalize(pts3d)
        phi = asin(ray[1, :])
        sign = cos(phi); sign[sign < 0] = -1.0; sign[sign >= 0] = 1.0
        theta = atan2(ray[2,:] * sign, -ray[0,:] * sign)
        equi_x = ((theta - np.pi / 2) / np.pi + 1) * w / 2
        equi_x[equi_x < 0] += w
        equi_x[equi_x >= w] -= w
        med = np.deg2rad((self.phi_max_deg - self.phi_min_deg) / 2)
        med2 = np.deg2rad((self.phi_max_deg + self.phi_min_deg) / 2)
        equi_y = ((phi + med2) / med + 1) * h / 2
        pix = stack([equi_x, equi_y], axis=0)
        pix[isnan(pix)] = -2
        return pix

    @staticmethod
    def create_equirect_camera(
            width: int, height: int,
            phi_deg: Optional[Union[float, Tuple[float, float]]] = None):
        if phi_deg is None:
            phi = height / width * 180.0
            phi_deg = (-phi, phi)
        elif type(phi_deg) != tuple:
            phi = abs(phi_deg)
            phi_deg = (-phi, phi)
        cam = EquirectCamModel()
        cam.width, cam.height = width, height
        cam.phi_min_deg = phi_deg[0]
        cam.phi_max_deg = phi_deg[1]
        return cam

class OrthographicCamModel(CameraModel):
    def __init__(self):
        super().__init__()
        self.h_size: float = 0
        self.v_size: float = 0
        self.type = CameraType.ORTHO

    def load_from_dict(self, dict: Dict[str, Any]) -> bool:
        if not super().load_from_dict(dict): return False
        try:
            self.h_size, self.v_size = dict['box_size']
            return True
        except KeyError as e:
            LOG_ERROR(e)
            return False

    def export_to_dict(self) -> Dict[str, Any]:
        dict = super().export_to_dict()
        dict['box'] = [self.h_size, self.v_size]
        return dict

    def pixel2ray(self, pts2d: ArrayType) -> ArrayType:
        rays = np.tile(np.array([0, 0, 1])[:, None], [1, pts2d.shape[1]])
        if is_torch_tensor(pts2d):
            rays = torch.tensor(rays, pts2d.dtype, pts2d.device)
        return rays

    def ray2pixel(self, pts3d: ArrayType) -> ArrayType:
        h, w = self.height, self.width
        xc = (w - 1) / 2.0
        yc = (h - 1) / 2.0
        xs = pts3d[0, :] / self.h_size * xc + xc
        ys = pts3d[1, :] / self.v_size * yc + yc
        return stack([xs, ys], axis=0)

    def pixel2origin(self, pts2d: ArrayType) -> ArrayType:
        h, w = self.height, self.width
        xc = (w - 1) / 2.0
        yc = (h - 1) / 2.0
        xs = (pts2d[0, :] - xc) / xc * self.h_size
        ys = (pts2d[1, :] - yc) / yc * self.v_size
        zs = zeros_like(xs)
        origins = stack([xs, ys, zs], axis=0)
        return origins

    def getRayOrigins(self) -> ArrayType:
        pts2d = self.pixel_grid()
        return self.pixel2origin(pts2d)

    @staticmethod
    def create_orthographic_camera(
            width: int, height: int,
            box_size: Union[float, Tuple[float, float]]):
        cam = OrthographicCamModel()
        cam.width = width
        cam.height = height
        if is_scalar_number(box_size):
            cam.h_size = box_size
            cam.v_size = cam.h_size * height / float(width)
        else:
            cam.h_size = box_size[0]
            cam.v_size = box_size[1]
        return cam

def load_camera_list_from_yaml(path: str) -> List[CameraModel]:
    config = yaml.safe_load(open(path))
    cams = []
    is_capture = 'sensor_nodes' in config.keys()
    if is_capture:
        nodes = config['sensor_nodes']
        for n in nodes:
            type_str = config[n]['type']
            if type_str == 'camera_multi_files' or type_str == 'multi_camera':
                cam_node_name = n
                break
        cam_node = config[cam_node_name]
        cam_keys = cam_node['sensor_nodes']
        calib_splitted = 'calib' in config.keys()
        for k in cam_keys:
            if calib_splitted:
                val = config['calib'][cam_node_name][k]
            else:
                val = cam_node[k]['calib']
            cam = CameraModel.create_from_dict(val)
            cams.append(cam)
    else:
        cameras_cfg = config['cameras']
        for i in range(len(cameras_cfg)):
            cam = CameraModel.create_from_dict(cameras_cfg[i])
            cams.append(cam)
    morph_filter = np.ones((5, 5), dtype=np.uint8)
    for cam in cams:
        invalid_mask = None
        yaml_dir, _ = osp.split(path)
        if cam.invalid_mask_file is not None:
            mask_file = osp.join(yaml_dir, cam.invalid_mask_file)
            if osp.exists(mask_file):
                invalid_mask = read_image(mask_file).astype(np.bool)
        cam.invalid_mask = cam.make_fov_mask(invalid_mask)
        cam.invalid_mask = ndimage.binary_closing(
            cam.invalid_mask, morph_filter, border_value=1)
    return cams

def load_camera_list_from_dict(config: Dict[str, Any]) -> List[CameraModel]:
    cams = []
    nodes = config['sensor_nodes']
    for n in nodes:
        type_str = config[n]['type']
        if type_str == 'camera_multi_files' or type_str == 'multi_camera':
            cam_node_name = n
            break
    cam_node = config[cam_node_name]
    cam_keys = cam_node['sensor_nodes']
    calib_splitted = 'calib' in config.keys()
    for k in cam_keys:
        if calib_splitted:
            val = config['calib'][cam_node_name][k]
        else:
            val = cam_node[k]['calib']
        cam = CameraModel.create_from_dict(val)
        cams.append(cam)
    morph_filter = np.ones((5, 5), dtype=np.uint8)
    yaml_dir = config['_yaml_path']
    for cam in cams:
        invalid_mask = None
        if cam.invalid_mask_file is not None:
            mask_file = osp.join(yaml_dir, cam.invalid_mask_file)
            if osp.exists(mask_file):
                invalid_mask = read_image(mask_file).astype(np.bool)
        cam.invalid_mask = cam.make_fov_mask(invalid_mask)
        cam.invalid_mask = ndimage.binary_closing(
            cam.invalid_mask, morph_filter, border_value=1)
    return cams

def load_camera_list_from_yaml_file(path: str) -> Union[List[CameraModel], None]:
    if not osp.exists(path):
        LOG_ERROR('YAML file does not exist: ' + path)
        return None
    config = yaml.safe_load(open(path))
    yaml_dir = osp.split(path)[0]
    config['_yaml_path'] = yaml_dir
    return load_camera_list_from_dict(config)
