import torch
import functools
import numpy as np
import itertools
import collections
import torch.sparse
from .MCOpBasics import _AddMCN, _ScalingN


def _Dot_MCN(x_tensor, y):
    """
    Dot product between tensor of a MCTensor(x_tensor) and a torch tensor (y),
    produce a sclar
    """
    assert x_tensor.size(0) == y.size(0)
    scaled_x_tensor = _ScalingN(x_tensor, y)
    # this could be much faster:
    result = torch.sum(scaled_x_tensor[..., 0])
    # slower version:
#     tmp = scaled_x_tensor[0:1]
#     # room for parallelism
#     for i in range(1, x_tensor.size(0)):
#         tmp = _AddMCN(tmp, scaled_x_tensor[i:i+1])
#     result = tmp[0, 0]
    return result


def _MV_MC_T(x_tensor, y):
    scaled = _ScalingN(x_tensor, y, style='V')
    tmp = scaled[..., 0, :]
    for i in range(1, scaled.size(-2)):
        tmp = _AddMCN(tmp, scaled[..., i, :])
    return tmp


def _MV_T_MC(x, y_tensor):
    scaled = _ScalingN(y_tensor, x, style='V')
    tmp = scaled[..., 0, :]
    for i in range(1, scaled.size(-2)):
        tmp = _AddMCN(tmp, scaled[..., i, :])
    return tmp


def _MM_MC_T(x_tensor, y):
    scaled = _ScalingN(x_tensor, y, style='MC-T')
    tmp = scaled[..., 0, :]
    for i in range(1, scaled.size(-2)):
        tmp = _AddMCN(tmp, scaled[..., i, :])
    return tmp


def _MM_T_MC(x, y_tensor):
    scaled = _ScalingN(x, y_tensor, style='T-MC')
    tmp = scaled[..., 0, :]
    for i in range(1, scaled.size(-2)):
        tmp = _AddMCN(tmp, scaled[..., i, :])
    return tmp


def _BMM_MC_T(x_tensor, y):
    x1, x2, _, nc = x_tensor.size()
    y1, _, y3 = y.size()
    size = max(x1,y1), x2, y3
    scaled = _ScalingN(x_tensor, y, style='BMM-MC-T')
    tmp = scaled[..., 0, :]
    for i in range(1, scaled.size(-2)):
        tmp = _AddMCN(tmp, scaled[..., i, :])
    return tmp, size, nc

def _BMM_T_MC(x, y_tensor):
    x1, x2, _ = x.size()
    y1, _, y3, nc = y_tensor.size()
    size = max(x1,y1), x2, y3
    scaled = _ScalingN(x, y_tensor, style='BMM-T-MC')
    tmp = scaled[..., 0, :]
    for i in range(1, scaled.size(-2)):
        tmp = _AddMCN(tmp, scaled[..., i, :])
    return tmp, size, nc

def _4DMM_MC_T(x_tensor, y):
    x1, x2, x3, _, nc = x_tensor.size()
    y1, y2, _, y3 = y.size()
    size = max(x1,y1), max(x2,y2), x3, y3
    scaled = _ScalingN(x_tensor, y, style='4DMM-MC-T')
    tmp = scaled[..., 0, :]
    for i in range(1, scaled.size(-2)):
        tmp = _AddMCN(tmp, scaled[..., i, :])
    return tmp, size, nc

def _4DMM_T_MC(x, y_tensor):
    x1, x2, x3, _ = x.size()
    y1, y2, _, y3, nc = y_tensor.size()
    size = max(x1,y1), max(x2,y2), x3, y3
    scaled = _ScalingN(x, y_tensor, style='4DMM-T-MC')
    tmp = scaled[..., 0, :]
    for i in range(1, scaled.size(-2)):
        tmp = _AddMCN(tmp, scaled[..., i, :])
    return tmp, size, nc


# def _broadcast_cartesian_product(x_size, y_size):
#     assert len(x_size) == len(y_size), "dim mismatch!"
#     x_ret, y_ret = [], []
#     correct_size = []
#     for i in range(max(len(x_size), len(y_size)) - 1, -1, -1):
#         x = x_size[i]
#         y = y_size[i]
#         if x != y:
#             if x == 1:
#                 x_ret.append([0] * y)
#                 y_ret.append(range(y))
#                 correct_size.append(y)
#             elif y == 1:
#                 x_ret.append(range(x))
#                 y_ret.append([0] * x)
#                 correct_size.append(x)
#             else:
#                 raise AssertionError("dim not compatible!")
#         else:
#             x_ret.append(range(x))
#             y_ret.append(range(y))
#             correct_size.append(x)

#     return (correct_size[::-1], itertools.product(*x_ret[::-1]), itertools.product(*y_ret[::-1]))


# def _equal_dim_matmul(x, y, x_MC=True):
#     # either x is tensor from MCTensor or y is
#     dtype, device = x.dtype, x.device
#     if x_MC:
#         x_size = x.size()[:-1]
#         y_size = y.size()
#         nc = x.size(-1)
#     else:
#         x_size = x.size()
#         y_size = y.size()[:-1]
#         nc = y.size(-1)
#     assert x_size[-1] == y_size[-2], "shape mismatch!"
#     correct_size, x_prod, y_prod = _broadcast_cartesian_product(
#         x_size[:-2], y_size[:-2])
#     x_prod = x_prod if x_size[0] > 1 else [0, ] * y_size[0]
#     ind = 0
#     size = *correct_size, x_size[-2], y_size[-1]
#     if x_MC:
#         tmp = torch.zeros(*correct_size, x_size[-2], y_size[-1],
#                           nc, device=device, dtype=dtype)
#         for x_idx in x_prod:
#             y_idx = next(y_prod)
#             tmp[ind] = _MM_MC_T(x[x_idx], y[y_idx])
#             ind += 1
#     else:
#         tmp = torch.zeros(*correct_size, x_size[-2], y_size[-1],
#                           nc, device=device, dtype=dtype)
#         for x_idx in x_prod:
#             y_idx = next(y_prod)
#             tmp[ind] = _MM_T_MC(x[x_idx], y[y_idx])
#             ind += 1
#     return tmp, size, nc
