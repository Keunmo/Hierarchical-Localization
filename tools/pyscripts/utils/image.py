# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# utils.image
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

import tifffile
import skimage.io
import skimage.transform
from PIL import Image
if TORCH_FOUND:
    import torch.nn.functional as F

## visualize =================================
def color_map(colormap_name: str,
             arr: np.ndarray,
             min_v: float = None,
             max_v: float = None,
             alpha: float = None) -> np.ndarray:
    arr = to_numpy(arr).astype(np.float64).squeeze()
    if colormap_name == 'oliver': return color_map_oliver(arr, min_v, max_v)
    cmap = matplotlib.cm.get_cmap(colormap_name)
    if max_v is None: max_v = np.nanmax(arr)
    if min_v is None: min_v = np.nanmin(arr)
    arr[arr > max_v] = max_v
    arr[arr < min_v] = min_v
    arr = (arr - min_v) / (max_v - min_v)
    if alpha is None:
        out = cmap(arr)
        if len(out.shape) == 3:
            out = out[:, :, 0:3]
        elif len(out.shape) == 2:
            out = out[:, 0:3]
    else:
        alpha = min(max(alpha, 0), 1)
        out = cmap(arr, alpha=alpha)
    return np.round(255 * out).astype(np.uint8)

#
# code adapted from Oliver Woodford's sc.m
_CMAP_OLIVER = np.array(
    [[0,0,0,114], [0,0,1,185], [1,0,0,114], [1,0,1,174], [0,1,0,114],
     [0,1,1,185], [1,1,0,114], [1,1,1,0]]).astype(np.float64)
#
def color_map_oliver(arr: np.ndarray,
                   min_v: float = None,
                   max_v: float = None) -> np.ndarray:
    arr = to_numpy(arr).astype(np.float64).squeeze()
    height, width = arr.shape
    arr = arr.reshape([1, -1])
    if max_v is None: max_v = np.max(arr)
    if min_v is None: min_v = np.min(arr)
    arr[arr < min_v] = min_v
    arr[arr > max_v] = max_v
    arr = (arr - min_v) / (max_v - min_v)
    bins = _CMAP_OLIVER[:-1, 3]
    cbins = np.cumsum(bins)
    bins = bins / cbins[-1]
    cbins = cbins[:-1] / cbins[-1]
    ind = np.sum(
        np.tile(arr, [6, 1]) > \
        np.tile(np.reshape(cbins,[-1, 1]), [1, arr.size]), axis=0)
    ind[ind > 6] = 6
    bins = 1 / bins
    cbins = np.array([0.0] + cbins.tolist())
    arr = (arr - cbins[ind]) * bins[ind]
    arr = _CMAP_OLIVER[ind, :3] * np.tile(np.reshape(1 - arr,[-1, 1]),[1,3]) + \
        _CMAP_OLIVER[ind+1, :3] * np.tile(np.reshape(arr,[-1, 1]),[1,3])
    arr[arr < 0] = 0
    arr[arr > 1] = 1
    out = np.reshape(arr, [height, width, 3])
    out = np.round(255 * out).astype(np.uint8)
    return out

# openGL coordinate mapping
def get_normal_map(n: ArrayType, im_size: Tuple[int, int]) -> np.ndarray:
    n = to_numpy(n)
    r = (n[0, :] + 1) / 2.0 # (H,W)
    g = (-n[1, :] + 1) / 2.0
    b = (-n[2, :] + 1) / 2.0 
    nmap = np.stack((r, g, b), -1).reshape((im_size[0], im_size[1], 3))
    return nmap

## image transform =================================

def rgb2gray(I: np.ndarray, channel_wise_mean=True) -> np.ndarray:
    I = to_numpy(I)
    dtype = I.dtype
    I = I.astype(np.float64)
    if channel_wise_mean:
        return np.mean(I, axis=2).squeeze().astype(dtype)
    else:
        return np.dot(I[...,:3], [0.299, 0.587, 0.114]).astype(dtype)

def imrescale(image: np.ndarray, scale: float) -> np.ndarray:
    image = to_numpy(image)
    dtype = image.dtype
    multi_channel = len(image.shape) == 3
    out = skimage.transform.rescale(image, scale,
        multichannel=multi_channel, preserve_range=True)
    return out.astype(dtype)

imresize = skimage.transform.resize

def bilinear_interpolation_numpy(I: np.ndarray, grid: np.ndarray) -> np.ndarray:
    def __interp_ch(I, x0, x1, y0, y1, rx, ry, rx1, ry1, invalid):
        I = ry1 * (I[y0, x0] * rx1 + I[y0, x1] * rx) + \
            ry * (I[y1, x0] * rx1 + I[y1, x1] * rx)
        I[invalid] = 0
        return I

    org_dtype = I.dtype
    I = I.astype(np.float64).squeeze()
    multi_channels = len(I.shape) == 3
    if multi_channels:
        src_h, src_w = I[..., 0].shape
    else:
        src_h, src_w = I.shape

    xs = grid[..., 0]
    ys = grid[..., 1]
    target_h, target_w = xs.shape
    xs = (xs + 1) / 2.0 * (src_w - 1) # make -1 ~ 1 to 0 ~ w-1
    ys = (ys + 1) / 2.0 * (src_h - 1) # make -1 ~ 1 to 0 ~ h-1

    invalid = logical_or(xs < 0, xs >= src_w, ys < 0, ys >= src_h,
        isnan(xs), isnan(ys))

    xs[invalid] = 0
    ys[invalid] = 0

    x0 = np.floor(xs).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, src_h - 1)

    rx = xs - x0
    ry = ys - y0
    rx1 = 1 - rx
    ry1 = 1 - ry

    if multi_channels:
        chs = [
            __interp_ch(
                I[..., n], x0, x1, y0, y1,
                rx, ry, rx1, ry1, invalid)[..., np.newaxis] \
            for n in range(I.shape[2])]
        return concat(chs, 2)
    else:
        return __interp_ch(I, x0, x1, y0, y1, rx, ry, rx1, ry1, invalid)


def bilinear_interpolation(I: ArrayType, grid: ArrayType) -> ArrayType:
    if not TORCH_FOUND: return bilinear_interpolation_numpy(I, grid)
    istensor = is_torch_tensor(I)
    if not istensor: I = torch.tensor(I)
    I = I.float().squeeze().unsqueeze(0) # make 1 x C x H x W
    if len(I.shape) < 4 : # if 1D channel image
        I = I.unsqueeze(0)
    is_flipped = I.shape[3] == 3 and I.shape[1] != 3
    if is_flipped: I = I.permute((0, 3, 1, 2))
    if not is_torch_tensor(grid): grid = torch.tensor(grid)
    grid = grid.squeeze().unsqueeze(0).float().to(I.device) # make 1 x npts x 2
    out = F.grid_sample(
        I, grid, mode='bilinear', align_corners=True).squeeze()
    if is_flipped: out = out.permute((1, 2, 0))
    if not istensor: out = out.numpy()
    return out

def pixel2grid(
        pts: ArrayType,
        target_resolution: Tuple[int, int],
        source_resolution: Tuple[int, int]) -> ArrayType:
    h, w = target_resolution
    height, width = source_resolution
    xs = (pts[0,:]) / (width - 1) * 2 - 1
    ys = (pts[1,:]) / (height - 1) * 2 - 1
    xs = xs.reshape((h, w, 1))
    ys = ys.reshape((h, w, 1))
    return concat((xs, ys), 2)

def normalize_image(
        image: np.ndarray,
        mask: Union[np.ndarray, None] = None) -> np.ndarray:
    image = to_numpy(image)
    mask = to_numpy(mask) > 0
    def normalize_image_1d(image, mask):
        image = image.squeeze().astype(np.float32)
        if mask is not None: image[mask] = np.nan
        # normalize intensities
        image = (image - np.nanmean(image.flatten())) / \
            np.nanstd(image.flatten())
        if mask is not None: image[mask] = 0
        return image
    if len(image.shape) == 3 and image.shape[2] == 3:
        # return np.concatenate(
        #     [normalize_image_1d(image[:,:,i], mask)[..., np.newaxis]
        #         for i in range(3)], axis=2)
        image = image.squeeze().astype(np.float32)
        mask = np.tile(mask[..., np.newaxis], (1, 1, 3))
        if mask is not None: image[mask] = np.nan
        # normalize intensities
        image = (image - np.nanmean(image.flatten())) / \
            np.nanstd(image.flatten())
        if mask is not None: image[mask] = 0
        return image
    else:
        return normalize_image_1d(image, mask)

def concat_images(
        imgs: List[List[np.ndarray]],
        indices: Optional[List[List[int]]] = None) -> np.ndarray:
    '''
    imgs: List[List[numpy.ndarray]]
        inner list: horizontal / outer list: vertical
    indices: List[List[int]]
        image indices
    '''
    if indices is not None:
        imgs = [
            [imgs[idx] for idx in horizon_inds] for horizon_inds in indices]
    horizon_imgs: List[np.ndarray] = [np.concatenate(x, 1) for x in imgs]
    return np.concatenate(horizon_imgs, 0)

## image file I/O =================================

def write_image_float(
        image: np.ndarray, tiff_path: str,
        thumbnail: Union[np.ndarray, None] = None) -> None:
    image = to_numpy(image)
    with tifffile.TiffWriter(tiff_path) as tiff:
        if thumbnail is not None:
            if not thumbnail.dtype == np.uint8:
                thumbnail = thumbnail.astype(np.uint8)
            tiff.write(thumbnail, photometric='RGB',
                bitspersample=8)
        if not image.dtype == np.float32:
            image = image.astype(np.float32)
        tiff.write(image, photometric='MINISBLACK',
                bitspersample=32, compression='zlib')

def read_image_float(
        tiff_path: str, return_thumbnail: bool = False,
        read_or_die: bool = True) -> Union[np.ndarray, List[np.ndarray], None]:
    try:
        # multi_image = skimage.io.MultiImage(tiff_path)
        multi_image = tifffile.TiffFile(tiff_path)
        num_read_images = len(multi_image.pages)
        if num_read_images == 0:
            raise Exception('No images found.')
        elif num_read_images == 1:
            return multi_image.pages[0].asarray().squeeze()
        elif num_read_images == 2: # returns float, thumbnail
            multi_image = [x.asarray().squeeze() for x in multi_image.pages]
            
            if multi_image[0].dtype == np.uint8:
                if not return_thumbnail: return multi_image[1].squeeze()
                else: return multi_image[1].squeeze(), multi_image[0].squeeze()
            else:
                if not return_thumbnail: return multi_image[0].squeeze()
                else: return multi_image[0].squeeze(), multi_image[1].squeeze()
        else: # returns list of images
            return [im.squeeze() for im in multi_image]
    except Exception as e:
        LOG_ERROR('Failed to read image float: "%s"' %(e))
        if read_or_die:
            traceback.print_tb(e.__traceback__)
            sys.exit()
        return None

def read_image_optional(path: str, cmap='magma', min_map=None, max_map=None, tiff_invdepth=True, read_or_die: bool = True):
    '''
        if png, call read_image.
        if tiff, call read_image_float and apply the colormap.
    '''
    extension :str = path.split('.')[-1]
    if extension == 'png':
        image_map = read_image(path, read_or_die=read_or_die)
    elif extension == 'tiff':
        image_float = read_image_float(path, read_or_die=read_or_die)
        if tiff_invdepth:
            if min_map is not None: min_map=1./min_map
            if max_map is not None: max_map=1./max_map
            image_map = color_map(cmap, 1.0 / image_float, min_map, max_map)
        else:
            image_map = color_map(cmap, image_float, min_map, max_map)
    else:
        LOG_FATAL(f".{extension} is not supported")
    return image_map

def write_image(image: np.ndarray, path: str) -> None:
    if image.dtype != np.uint16:
        image = skimage.img_as_ubyte(to_numpy(image).astype(np.uint8))
    image = Image.fromarray(image)
    # skimage.io.imsave(path, image, check_contrast=False)
    image.save(path)

def read_image(path: str, read_or_die = True) -> Union[np.ndarray, None]:
    try:
        return skimage.io.imread(path)
    except Exception as e:
        LOG_ERROR('Failed to read image: "%s"' % (e))
        if read_or_die:
            traceback.print_tb(e.__traceback__)
            sys.exit()
        return None

def read_image_float3(
        tiff_path: str, return_thumbnail: bool = False,
        read_or_die: bool = True) -> Union[np.ndarray, List[np.ndarray], None]:
    try:
        multi_image = tifffile.TiffFile(tiff_path)
        num_read_images = len(multi_image.pages)
        if num_read_images == 0:
            raise Exception('No images found.')
        elif num_read_images == 3: # return float3
            return np.concatenate([x.asarray()[np.newaxis,:,:] for x in multi_image.pages], axis=0) 
        elif num_read_images == 4: # return float3, thumbnail
             image = [x.asarray()[np.newaxis,:,:] for x in multi_image.pages[1:]]
             image = np.concatenate(image, axis = 0)
             thumbnail = multi_image.pages[0].asarray()
             if return_thumbnail:
                 return image, thumbnail
             else:
                 return image
        else:
            raise Exception('invalid type.') 
    except Exception as e:
        LOG_ERROR('Failed to read image float3: "%s"' %(e))
        if read_or_die:
            traceback.print_tb(e.__traceback__)
            sys.exit()
        return None        

