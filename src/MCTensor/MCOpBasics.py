import torch
import functools
import numpy as np
import itertools
import collections
import torch.sparse


def Two_Sum(a, b):
    """
    performs two-sum addition of two standard float vectors
    """
    x = a + b
    b_virtual = x - a
    a_virtual = x - b_virtual
    b_roundoff = b - b_virtual
    a_roundoff = a - a_virtual
    y = a_roundoff + b_roundoff
    return x, y


def QTwo_Sum(a, b):
    """
    performs quick-two-sum addition of two standard float vectors, work for n dimensional vectors, and b*n dimension
    """
    s = a + b
    e = b - (s - a)
    return s, e


def Split(a):
    """
    The following algorithm splits a 53-bit IEEE double precision floating point
    number into ahi and alo, each with 26 bits of significand,
    such that a = ahi + alo. ahi will contain the first 26 bits, while alo will contain the lower 26 bits.
    26 -> 12 for float iirc
    53 bit double and 24bit single, i.e. 26. and 12 for the constant
    """
    if a.dtype == torch.float16:
        constant = 6
    elif a.dtype == torch.float32:
        constant = 12
    elif a.dtype == torch.float64:
        constant = 26
    else:
        raise NotImplemented
    t = (2**constant+1)*a
    ahi = t-(t-a)
    alo = a - ahi
    return ahi, alo


def Two_Prod(a, b):
    """
    performs two-prod algorithm, computes p = fl(a × b) and e = err(a × b).
    """
    p = a * b
    ahi, alo = Split(a)
    bhi, blo = Split(b)
    e = ((ahi*bhi-p)+ahi*blo+alo*bhi)+alo*blo
    # """
    # performs two-prod-fma algorithm, computes p = fl(a × b) and e = err(a × b).
    # """
    # p = a * b
    # e = torch.addcmul(-p, a, b)
    return p, e


def _Renormalize(x_tensor, r_nc=None):
    """
    x_tensor: tensor of a MCTensor to be renormalized
    r_nc: reduced number of components, <= nc of x
    """
    nc = x_tensor.size(-1)
    if r_nc is None:
        r_nc = nc - 1
    rough_x = torch.zeros_like(x_tensor)
    s = x_tensor.data[..., 0]  # the first (largest) component
    # first for loop, two sum from large to small components
    for i in range(1, nc):
        # two sum of s and i component
        s, rough_x_val = Two_Sum(x_tensor.data[..., i], s)
        rough_x[..., i] = rough_x_val  # components in decreasing order now
    # note here s is the smallest component, there are nc-1 components in rough_x
    normalized_x = torch.zeros_like(x_tensor.data)
    # second for loop
    for i in range(nc-1):
        s, e = Two_Sum(s, rough_x[..., i])
        # the following aims to handle the if condition, but we may need to further discuss it
        nonzero_ind_e = torch.nonzero(e, as_tuple=True)
        nonzero_ind_x = tuple(
            list(nonzero_ind_e) + [-(1+i) * torch.ones(len(nonzero_ind_e[0])).long()])
        # maybe needs to switch following behavior with copy, and make sure the data is changed
        normalized_x[nonzero_ind_x] = s[nonzero_ind_e]
        s[nonzero_ind_e] = e[nonzero_ind_e]
    normalized_x[..., 0] = s
    # as of now, the components in normalized_x may not be correctly ordered, so we sort it according to
    # absolute values
    _, indices = torch.sort(torch.abs(normalized_x), descending=True)
    # fill this tensor in as contents for MCTensor
    normalized_x = torch.gather(normalized_x, -1, indices)[..., :r_nc]
    return normalized_x


def _Simple_renormalize(tensor_list, r_nc):
    all_tensor = torch.cat(
        [tensor_list[0].data, tensor_list[1].data.unsqueeze(-1)], dim=-1)
    _, indices = torch.sort(torch.abs(all_tensor.data), descending=True)
    # fill this tensor in as contents for MCTensor
    result = torch.gather(all_tensor.data, -1, indices)[..., :r_nc]
    return result

# def _Simple_renormalize(tensor_list, r_nc):
#     ## inplace operations
#     tensors = torch.cat(
#         [tensor_list[0].data, tensor_list[1].data.unsqueeze(-1)], dim=-1)
#     *indice_wolast, indice_last = torch.nonzero(tensors, as_tuple=True)
#     indice_wolast_list = [indice_wolast[i].tolist() for i in range(len(indice_wolast))]
#     indice_wolast_list = zip(*indice_wolast_list)
#     number_of_uniques = list(collections.Counter(indice_wolast_list).values())
#     indice_last_new = [list(range(number_of_uniques[i])) for i in range(len(number_of_uniques))]
#     indice_last_new = torch.tensor(sum(indice_last_new, []))
#     tensors_clone = torch.zeros_like(tensors)
#     tensors_clone.data[indice_wolast + [indice_last_new]] = tensors.data[indice_wolast + [indice_last]]
#     return tensors_clone[..., :r_nc]


def _Grow_ExpN(x_tensor, value, simple=True):
    nc = x_tensor.size(-1)
    Q = value
    h = torch.zeros_like(x_tensor)
    for i in range(1, nc+1):
        Q, hval = Two_Sum(x_tensor[..., -i], Q)
        if i == 1:
            last_tensor = hval.data
        else:
            h[..., -(i-1)] = hval.data
    h[..., 0] = Q
    if simple:
        h.data.copy_(_Simple_renormalize([h, last_tensor], r_nc=nc))
    else:
        h.data.copy_(_Renormalize(
            torch.cat([h.data, last_tensor.data.unsqueeze(-1)], dim=-1), r_nc=nc))
    return h


def _AddMCN(x_tensor, y_tensor, simple=True):
    nc = x_tensor.size(-1)
    h = torch.zeros_like(x_tensor)
    e = torch.tensor(0)  # since two_sum does the conversion to tensor
    for i in range(nc):
        hp, e1 = Two_Sum(x_tensor[..., i], y_tensor[..., i])
        hi, e2 = Two_Sum(hp, e)
        h_to_append = hi if i == 0 else hi.data
        h[..., i] = h_to_append
        e = e1 + e2
    if simple:
        h.data.copy_(_Simple_renormalize([h, e], r_nc=nc))
    else:
        h.data.copy_(_Renormalize(
            torch.cat([h.data, e.data.unsqueeze(-1)], dim=-1), r_nc=nc))
    return h


def _ScalingN(x_tensor, value, style='V', expand=False, simple=True):
    if style in ['T-MC', 'BMM-T-MC', '4DMM-T-MC']:
        nc = value.size(-1)
    else:
        nc = x_tensor.size(-1)
    e = torch.tensor(0)
    for i in range(nc):
        if style == 'V':
            hval_pre, e1 = Two_Prod(x_tensor[..., i], value)
        elif style == 'EXPM':
            hval_pre, e1 = Two_Prod(x_tensor[..., i].unsqueeze(-1), value)
        elif style == 'MC-T':
            hval_pre, e1 = Two_Prod(
                x_tensor[..., i].unsqueeze(-2), value.transpose(-1, -2))
        elif style == 'T-MC':
            hval_pre, e1 = Two_Prod(
                x_tensor.unsqueeze(-2), value[..., i].transpose(-1, -2))
        elif style == 'BMM-MC-T':
            hval_pre, e1 = Two_Prod(
                x_tensor[..., i].unsqueeze(-2), value.unsqueeze(1).transpose(-1, -2))
        elif style == 'BMM-T-MC':
            hval_pre, e1 = Two_Prod(
                x_tensor.unsqueeze(-2), value[..., i].unsqueeze(1).transpose(-1, -2))
        elif style == '4DMM-MC-T':
            hval_pre, e1 = Two_Prod(
                x_tensor[..., i].unsqueeze(-2), value.unsqueeze(2).transpose(-1, -2))
        elif style == '4DMM-T-MC':
            hval_pre, e1 = Two_Prod(
                x_tensor.unsqueeze(-2), value[..., i].unsqueeze(2).transpose(-1, -2))
        else:
            raise NotImplementedError()
        hval, e2 = Two_Sum(hval_pre, e)
        if i == 0:
            h_to_append = hval
            h = torch.zeros(h_to_append.size() + (nc,)).to(h_to_append)
        else:
            h_to_append = hval.data
        h[..., i] = h_to_append
        e = e1 + e2
    if expand:
        r_nc = nc + 1
    else:
        r_nc = nc
    if simple:
        rh = _Simple_renormalize([h, e], r_nc=r_nc)
    else:
        rh = _Renormalize(
            torch.cat([h.data, e.data.unsqueeze(-1)], dim=-1), r_nc=r_nc)
    h.data.copy_(rh.data[..., :nc])
    if expand:
        h = torch.cat([h, rh.data[..., -1:]], dim=-1)
    return h


def _DivMCN(x_tensor, y_tensor, simple=False):
    nc = y_tensor.size(-1)
    h = torch.zeros_like(x_tensor)
    # approx quotient q0
    q = x_tensor[..., 0] / y_tensor[..., 0]
    h[..., 0] = q
    for i in range(1, nc+1):
        r = _AddMCN(x_tensor, -_ScalingN(y_tensor, q),
                    simple=True)  # r = x - y*q
        x_tensor = r
        q = x_tensor.data[..., 0] / y_tensor.data[..., 0]
        if i != nc:
            h[..., i] = q
    if simple:
        h.data.copy_(_Simple_renormalize([h, q], r_nc=nc))
    else:
        h.data.copy_(_Renormalize(
            torch.cat([h.data, q.data.unsqueeze(-1)], dim=-1), r_nc=nc))
    return h


def _MultMCN(x_tensor, y_tensor, simple=False):
    # this might be faster, but worse error bounds
    nc = x_tensor.size(-1)
    ones_like_y = torch.ones_like(y_tensor)
    ones_like_y[..., 1:].zero_()
    y_inv = _DivMCN(ones_like_y, y_tensor, simple=simple)
    result = _DivMCN(x_tensor, y_inv, simple=simple)
    return result

# def _MultMCN(x_tensor, y_tensor, simple=True):
#     # this is slower, but with better error guarantee
#     nc = x_tensor.size(-1)
#     h = torch.zeros_like(x_tensor)
#     # approx quotient q0
#     p = x_tensor[..., 0] * y_tensor[..., 0]
#     h[...,0] = p
#     # convert p to MCTensor like
#     p_MC_tensor = torch.zeros(p.size()+(nc,)).to(x_tensor)
#     p_MC_tensor[..., 0] = p.data
#     for i in range(1, nc+1):
#         # if replace this with an algo similar to scalingN, this would be much faster
#         minus_pdivy = - _DivMCN(p_MC_tensor, y_tensor, simple=simple)
#         e = _AddMCN(x_tensor, minus_pdivy, simple=simple)
#         x_tensor = e
#         p = x_tensor.data[..., 0] * y_tensor.data[..., 0]
#         if i!=nc: h[..., i] = p
#         p_MC_tensor.zero_()
#         p_MC_tensor[..., 0] = p.data
#     if simple:
#         h.data.copy_(_Simple_renormalize([h, p], r_nc = nc))
#     else:
#         h.data.copy_(_Renormalize(torch.cat([h.data, p.data.unsqueeze(-1)], dim=-1), r_nc = nc))
#     return h


def _exp(input_tensor):
    nc = input_tensor.size(-1)
    MCF_exp = torch.exp(input_tensor)
    tmp = MCF_exp[..., 0:1]
    for i in range(1, nc):
        tmp = _ScalingN(tmp, MCF_exp[..., i], expand=True)
    return tmp


def _square(input_tensor):
    # approx square with (x1…xn)^2 = x1^2 + 2x1x2
    nc = input_tensor.size(-1)
    x1 = input_tensor[..., 0]
    if nc == 1:
        return torch.square(x1)
    tmp = torch.zeros_like(input_tensor)
    x2 = input_tensor[..., 1]
    tmp[..., 0] = torch.square(x1)
    return _Grow_ExpN(tmp, (2*x1*x2).data)


def _pow(input_tensor, exponent):
    raise "Power currently wrong"
    # nc = input_tensor.size(-1)
    # MCF_pow = torch.pow(input_tensor, exponent)
    # tmp = MCF_pow[..., 0:1]
    # for i in range(1, nc):
    #     tmp = _ScalingN(tmp, MCF_pow[..., i], expand=True)
    # return tmp
    