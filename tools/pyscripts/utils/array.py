# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# utils.array.py
#
# Author: Changhee Won (changhee.won@multipleye.co)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import *

from utils.common import *

if TORCH_FOUND:
    ArrayType = Union[np.ndarray, torch.Tensor]
    TensorList = List[torch.Tensor]
else:
    ArrayType = np.ndarray

def is_scalar_number(x: Any):
    return (type(x) == float or type(x) == int or nelem(x) == 1)

def to_numpy(
        arr: Union[List, ArrayType], dtype: Optional[np.dtype] = None) -> \
            np.ndarray:
    if is_torch_tensor(arr): arr = arr.cpu().numpy()
    arr = np.array(arr)
    if dtype is not None: arr = arr.astype(dtype)
    return arr

def to_torch_as(arr: np.ndarray, tensor: 'torch.Tensor') -> 'torch.Tensor':
    if not TORCH_FOUND:
        LOG_ERROR('to_torch_as: torch module not found')
        return arr
    if is_torch_tensor(arr):
        return arr.to(tensor.dtype).to(tensor.device)
    else:
        return torch.tensor(arr, dtype=tensor.dtype, device=tensor.device)

def clone(arr: ArrayType) -> ArrayType:
    if is_torch_tensor(arr): return arr.clone()
    else: return arr.copy()

def norm(x: ArrayType) -> Union[ArrayType, float]:
    n = sqrt((x**2).sum(0))
    if nelem(n) == 1:
        n = n.flatten()[0]
    return n

def sqrt(x: ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.sqrt(x)
    else: return np.sqrt(x)

def normalize(x: ArrayType) -> ArrayType:
    n = norm(x) + EPS_SINGLE
    return x / n

def normalize_by_z(x: ArrayType) -> ArrayType:
    n = x[2,:]
    return x / n

def cos(x: ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.cos(x)
    else: return np.cos(x)

def sin(x: ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.sin(x)
    else: return np.sin(x)

def tan(x: ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.tan(x)
    else: return np.tan(x)

def atan(x:ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.atan(x)
    else: return np.arctan(x)

def atan2(y: ArrayType, x:ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.atan2(y, x)
    else: return np.arctan2(y, x)

def asin(x: ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.asin(x)
    else: return np.arcsin(x)

def acos(x: ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.acos(x)
    else: return np.arccos(x)

def exp(x: ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.exp(x)
    else: return np.exp(x)

def reshape(x: ArrayType, shape: List[int]) \
        -> ArrayType:
    if is_torch_tensor(x): return x.view(shape)
    else: return x.reshape(shape)

def concat(arr_list: List[ArrayType], axis: int = 0, contiguous: bool = True):
    if is_torch_tensor(arr_list[0]):
        out = torch.cat(arr_list, dim=axis)
        if contiguous: out = out.contiguous()
        return out
    else:
        out = np.concatenate(arr_list, axis=axis)
        if contiguous: out = np.ascontiguousarray(out)
        return out

def stack(arr_list: List[ArrayType], axis: int = 0):
    if is_torch_tensor(arr_list[0]): return torch.stack(arr_list, axis)
    else: return np.stack(arr_list, axis)

def polyval(P: ArrayType, x: ArrayType) -> ArrayType:
    '''
        P: pol or invpol coefficients
        x: theta or r, respectively
        return: r or theta, respectively
    '''
    if is_torch_tensor(x): P = torch.tensor(P).to(x.device) # if x is tensor, convert P to tensor as well
    if is_torch_tensor(P): # if tensor, from scratch. 
        npol = P.shape[0] # num of coefficient
        val = torch.zeros_like(x)
        for i in range(npol - 1):
            val = (val + P[i]) * x
        val += P[-1] # don't want the constant term
        return val
    else: # use numpy function if a ndarray is given
        return np.polyval(P, to_numpy(x))
        #p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1] # polyval()

def isnan(arr: ArrayType):
    if is_torch_tensor(arr): return torch.isnan(arr)
    else: return np.isnan(arr)

def zeros_like(x: ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.zeros_like(x)
    else: return np.zeros_like(x)

def ones_like(x: ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.ones_like(x)
    else: return np.ones_like(x)

def empty_like(x: ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.empty_like(x)
    else: return np.empty_like(x)

def where(cond: ArrayType, x: ArrayType, y: ArrayType) -> ArrayType:
    if is_torch_tensor(cond): return torch.where(cond, x, y)
    else: return np.where(cond, x, y)

def logical_and(*arrs: List[ArrayType]) -> ArrayType:
    if len(arrs) == 2:
        if is_torch_tensor(arrs[0]):
            return torch.logical_and(arrs[0], arrs[1].reshape(arrs[0].shape))
        else:
            return np.logical_and(arrs[0], arrs[1].reshape(arrs[0].shape))
    return logical_and(arrs[0], logical_and(*arrs[1:]))

def logical_or(*arrs: List[ArrayType]) -> ArrayType:
    if len(arrs) == 2:
        if is_torch_tensor(arrs[0]):
            return torch.logical_or(arrs[0], arrs[1].reshape(arrs[0].shape))
        else:
            return np.logical_or(arrs[0], arrs[1].reshape(arrs[0].shape))
    return logical_or(arrs[0], logical_or(*arrs[1:]))

def logical_not(arr: ArrayType) -> ArrayType:
    if is_torch_tensor(arr): return torch.logical_not(arr)
    else: return np.logical_not(arr)

def hom(arr: ArrayType) -> ArrayType:
    h = ones_like(arr[0, :].reshape((1, -1)))
    return concat((arr, h), 0)

def nelem(arr: ArrayType) -> int:
    if is_torch_tensor(arr): return arr.nelement()
    else: return arr.size

def polyfit(
        x: np.ndarray, y: np.ndarray, deg: int,
        zero_coeffs: Optional[List[int]] = None) -> np.ndarray:
    if zero_coeffs is None: return np.polyfit(x, y, deg)
    x = x.flatten()
    y = y.reshape((-1, 1))
    ndata = x.shape[0]
    V = np.zeros((ndata, deg + 1), dtype=np.float64)
    V[:,-1] = 1.0
    for n in range(deg - 1, -1, -1):
        V[:, n] = x * V[:, n + 1]
    zero_coeffs = [deg - x for x in zero_coeffs]
    non_zero = list(set(list(range(deg + 1))) - set(zero_coeffs))
    V = V[:, non_zero]
    Q, R = np.linalg.qr(V)
    p = np.linalg.inv(R) @ (Q.T @ y)
    p_ = np.zeros((deg + 1), dtype=np.float64)
    p_[non_zero] = p.flatten()
    p_[zero_coeffs] = 0.0
    return p_

def gather(x: ArrayType, axis: int, index: ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.gather(x, axis, index)
    else: return np.take_along_axis(x, index, axis)

def tile(x: ArrayType, reps: List[int]) -> ArrayType:
    if is_torch_tensor(x): return torch.tile(x, reps)
    else: return np.tile(x, reps)

def maximum(x: ArrayType, y: ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.maximum(x, y)
    else: return np.maximum(x, y)

def minimum(x: ArrayType, y: ArrayType) -> ArrayType:
    if is_torch_tensor(x): return torch.minimum(x, y)
    else: return np.minimum(x, y)

def roll(x: ArrayType, shifts: int, axis: int) -> ArrayType:
    if is_torch_tensor(x): return torch.roll(x, shifts, axis)
    else: return np.roll(x, shifts, axis)

def cross(x: ArrayType, y: ArrayType, axis: Optional[int] = None) -> ArrayType:
    if is_torch_tensor(x): return torch.cross(x, y, axis)
    else: return np.cross(x, y, axis=axis)
