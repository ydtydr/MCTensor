import torch
import functools
import numpy as np
import itertools
import collections
import torch.sparse
import random

from .MCOpBasics import _Renormalize, _Grow_ExpN, _AddMCN,  _ScalingN,\
    _DivMCN, _MultMCN, _exp, _pow, _square
from .MCOpMatrix import _Dot_MCN, _MV_MC_T, _MV_T_MC, _MM_MC_T, _MM_T_MC,\
    _BMM_MC_T, _BMM_T_MC, _4DMM_T_MC, _4DMM_MC_T


HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override for MCTensor"""
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator


class MCTensor(object):
    def __init__(self, *size, nc, val=None,
                 requires_grad=False, device=None,
                 sparse_grad=False, dtype=None):
        """ size: size of the mc tensor,
            nc: number of multicomponent. 
            val: initialization of the tensor, need to support both standard 
            float tensor and MCTensor also needs to support MCTensor.like function 
            support cuda
            Invariant: self.tensor is a MCF, with individual float componenet 
            x_0>x_1>...>
        """
        self.device = device
        self.dtype = dtype
        if isinstance(size[0], tuple):
            self._size = tuple(list(size[0]) + [nc])
        else:
            self._size = tuple(list(size) + [nc])
        self._nc = nc
        # change val to numpy if it's a list
        if isinstance(val, list):
            val = np.asarray(val)
        # val from the input value
        if val is None:
            # initialize randomly
            self.tensor = torch.randn(self._size, device=device, dtype=dtype)
            tmp = _Renormalize(self.tensor, r_nc=nc)
            self.fc = tmp[..., :1].clone().detach()
            self.fc.requires_grad_(requires_grad)
            self.tensor = torch.cat(
                [self.fc, tmp[..., 1:]], dim=-1)
        elif isinstance(val, MCTensor):
            if device == None:
                device = val.device
            if dtype == None:
                dtype = val.dtype
            # initialize from MCTensor
            assert val.size_nc() == self._size, 'init from MCTensor has inconsistent dimension!'
            if requires_grad:
                self.fc = val.tensor.data[..., :1].clone(
                ).detach().to(dtype).to(device)
                self.fc.requires_grad_(requires_grad)
            else:
                # keep track of the gradient graph
                self.fc = val.tensor[..., :1].to(dtype).to(device)
            self.tensor = torch.cat(
                [self.fc, val.tensor.data[..., 1:].to(dtype).to(device)], dim=-1)
        elif isinstance(val, (np.ndarray, np.generic)):
            # initialize from numpy array
            assert (tuple(val.shape) + (nc,)
                    ) == self._size, 'init from Numpy array has inconsistent dimension!'
            np_init = torch.from_numpy(val)
            if device == None:
                device = np_init.device
            if dtype == None:
                dtype = np_init.dtype
            self.fc = np_init.unsqueeze(
                -1).clone().detach().to(dtype).to(device)
            self.fc.requires_grad_(requires_grad)
            self.tensor = torch.cat([self.fc, torch.zeros(
                tuple(list(self._size[:-1]) + [nc-1]), device=device, dtype=dtype)], dim=-1)
        elif isinstance(val, torch.Tensor):
            if device == None:
                device = val.device
            if dtype == None:
                dtype = val.dtype

            # initialize from torch
            if tuple(val.size()) == self._size:
                if requires_grad:
                    self.fc = val.data[..., :1].clone(
                    )
                    self.fc.detach().to(dtype).to(device)
                    self.fc.requires_grad_(requires_grad)
                    self.tensor = torch.cat(
                        [self.fc, val.data[..., 1:].to(dtype).to(device)], dim=-1)
                else:
                    # keep track of the gradient graph
                    self.fc = val[..., :1].to(dtype).to(device)
                    self.tensor = val.to(dtype).to(device)
            elif (tuple(val.size()) + (nc,)) == self._size:
                if requires_grad:
                    self.fc = val.data.unsqueeze(
                        -1).clone().detach().to(dtype).to(device)
                    self.fc.requires_grad_(requires_grad)
                else:
                    # keep track of the gradient graph
                    self.fc = val.unsqueeze(-1).to(dtype).to(device)
                # need to be cuda compatible, zeros device
                self.tensor = torch.cat([self.fc, torch.zeros(
                    tuple(list(self._size[:-1]) + [nc-1]), device=device, dtype=dtype)], dim=-1)
            else:
                raise Exception(
                    'init from torch tensor has inconsistent dimension!', val.size(), self._size)
            if val.requires_grad:
                requires_grad = True
        else:
            raise Exception('The input data type is not supported!')

        self.device = self.tensor.device
        self.dtype = self.tensor.dtype
        self.is_sparse_grad = sparse_grad

    def __repr__(self):
        return "MCTensor(Size={}, number of components={}, requires_grad={})".format(self.size(), self.size_nc(-1), self.requires_grad)

    def __getitem__(self, key):
        if key[0] == Ellipsis:
            key = key + (slice(None, None, None), )
        tensor_slice = self.tensor[key]
        if tensor_slice.dim() == 1:
            tensor_slice = tensor_slice.unsqueeze(0)
        return MCTensor(tensor_slice.size()[:-1], nc=self.size_nc(-1), val=tensor_slice,
                        device=self.device, dtype=self.dtype)

    def __setitem__(self, key, value):
        if key[0] == Ellipsis:
            key = key + (slice(None, None, None), )
        if isinstance(value, torch.Tensor):
            if value.size() == self.tensor[key][..., 0].size():
                self.tensor[key].zero_()
                self.fc[key][...,0] = value
                self.tensor[key][..., 0] = value
            elif value.size() == self.tensor[key].size():
                self.fc[key][...,0] = value[..., 0]
                self.tensor[key] = value
        elif isinstance(value, MCTensor):
            self.fc[key][..., 0] = value.tensor[..., 0]
            self.tensor[key] = value.tensor
        else:
            raise NotImplemented

    def __len__(self):
        return len(self.tensor[..., 0])

    def __add__(self, other):
        ''' add self with other'''
        return torch.add(self, other)

    def __radd__(self, other):
        ''' add self with other'''
        return torch.add(other, self)

    def __sub__(self, other):
        ''' subtract self with other'''
        return torch.add(self, -other)

    def __rsub__(self, other):
        ''' subtract self from other'''
        return torch.add(other, -self)

    def __matmul__(self, other):
        '''matrix mul self with other '''
        return torch.matmul(self, other)

    def __mul__(self, other):
        ''' mul self with other'''
        return torch.mul(self, other)

    def __rmul__(self, other):
        ''' mul self with other'''
        return torch.mul(other, self)

    def __truediv__(self, other):
        ''' division self with other'''
        return torch.div(self, other)

    def __rtruediv__(self, other):
        ''' division self from other'''
        return torch.div(other, self)

    def __neg__(self):
        ''' negated self'''
        return MCTensor(self.size(), nc=self.size_nc(-1), val=-self.tensor,
                        device=self.device, dtype=self.dtype)

    def __pos__(self):
        ''' positive self'''
        return self

    def __abs__(self):
        ''' positive self'''
        return MCTensor(self.size(), nc=self.size_nc(-1), val=torch.abs(self.tensor), device=self.device)

    def __pow__(self, other):
        return torch.pow(self, other)

    def stride(self, *args):
        if len(args) == 0:
            return self.tensor.stride()[:-1]
        else:
            return self.tensor.stride(*args)

    def sum(self):
        return torch.sum(self.tensor[..., 0])

    @property
    def T(self):
        (*shape, nc) = self.size_nc()
        T_shape = shape[::-1]
        values = torch.zeros((*T_shape, nc), device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        for i in range(nc):
            values[..., i].data.copy_(self.tensor[..., i].T)
        return MCTensor(*T_shape, nc=nc, val=values, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)

    @property
    def grad(self):
        if self.fc.grad is not None:
            if not self.is_sparse_grad:
                return self.fc.grad[..., 0]
            else:
                return self.fc.grad[..., 0].to_sparse(sparse_dim=1)
        else:
            return None

    @property
    def data(self):
        return self.tensor.data
    
    @property
    def requires_grad(self):
        return self.fc.requires_grad
    
    @requires_grad.setter
    def requires_grad(self, value):
        self.fc.requires_grad = value

    @property
    def shape(self):
        return self.size()

    def size(self, *args, **kwargs):  # need to double check usage of size for MCTensor
        return self.tensor[..., 0].size(*args, **kwargs)

    def size_nc(self, *args, **kwargs):
        return self.tensor.size(*args, **kwargs)

    def matmul(self, other):
        return torch.matmul(self, other)

    def addmm(self, mat1, mat2, beta=1, alpha=1):
        return torch.addmm(self, mat1=mat1, mat2=mat2, beta=beta, alpha=alpha)

    def transpose(self, *args, **kws):
        return torch.transpose(self, *args, **kws)

    def dim(self):
        return self.tensor.dim() - 1
    
    def clamp_(self, min=None, max=None):   
        fc = self.fc.clamp_(min=min, max=max)
        self.tensor[..., :1] = fc
        if min is not None:
            cond = fc > min
        else:
            cond = fc < max
        self.data[..., 1:] = torch.where(cond, self.data[..., 1:], torch.zeros_like(fc))
        return self

    def clamp(self, min=None, max=None):
        new = self.clone()
        new.tensor.clamp_(min=min, max=max)
        return new

    def expand(self, *sizes):
        nc = self.size_nc(-1)
        if isinstance(sizes[0], tuple):
            _size_expand = sizes[0] + (nc,)
            size_new = sizes[0]
        else:
            _size_expand = sizes + (nc,)
            size_new = sizes
        expanded_t = self.tensor.expand(_size_expand)
        return MCTensor(size_new, nc=nc, val=expanded_t)

    def expand_as(self, other):
        return self.expand(other.size())

    def narrow(self, dim, start, length):
        assert dim < self.dim() and dim >= -self.dim(), f"Dimension out of range!"
        if dim < 0:
            dim = dim + self.dim()
        self_t = self.tensor
        narrowed = self_t.narrow(dim, start, length)
        narrow_size = narrowed.size()[:-1]
        return MCTensor(narrow_size, nc=self.size_nc(-1),
                        val=narrowed)

    def clone(self):
        new = MCTensor(self.size(), nc=self.size_nc(-1),
                       val=self.tensor.clone(), device=self.device)
        return new

    def detach(self):
        new = MCTensor(self.size(), nc=self.size_nc(-1),
                       val=self.tensor.detach(), device=self.device)
        return new

    def cuda(self):
        self_copy = MCTensor(
            self.size(), nc=self.size_nc(-1), val=self.tensor.clone().cuda(), device='cuda')
        return self_copy

    def to(self, device):
        self_copy = MCTensor(
            self.size(), nc=self.size_nc(-1), val=self.tensor.clone().to(device), device=device)
        return self_copy

    def to_type(self, dtype):
        self_copy = MCTensor(
            self.size(), nc=self.size_nc(-1), val=self.tensor.clone(),
            device=self.device, dtype=dtype)
        return self_copy

    def mul_(self, m):
        # if self.fc.is_leaf:
        #     raise RuntimeError("leaf node detected")
        copy = torch.mul(self, m)
        self.fc.data.copy_(copy.fc.data)
        self.tensor.data.copy_(copy.tensor.data)
        return self

    def add_(self, m, alpha=1):
        # if self.fc.is_leaf:
        #     raise RuntimeError("leaf node detected")
        copy = torch.add(self, m, alpha=alpha)
        # self.fc = copy.fc
        self.fc.data.copy_(copy.fc.data)
        # self.tensor = torch.cat([self.fc, copy.data[..., 1:]], dim=-1)
        self.tensor.data.copy_(copy.tensor.data)
        return self

    def addcmul_(self, tensor1, tensor2, value=1):
        # if self.fc.is_leaf:
        #     raise RuntimeError("leaf node detected")
        copy = torch.add(self, torch.mul(tensor1, tensor2), alpha=value)
        self.fc.data.copy_(copy.fc.data)
        self.tensor.data.copy_(copy.tensor.data)
        return self

    def addcdiv_(self, tensor1, tensor2, value=1):
        # if self.fc.is_leaf:
        #     raise RuntimeError("leaf node detected")
        copy = torch.add(self, torch.div(tensor1, tensor2), alpha=value)
        self.fc.data.copy_(copy.fc.data)
        self.tensor.data.copy_(copy.tensor.data)
        return self

    def sqrt(self):
        sign = torch.sign(self.tensor)
        value = torch.abs(self.tensor).sqrt()
        return MCTensor(self.size(), nc=self.size_nc(-1),
                        val=(value * sign), device=self.device)
        # value = torch.nan_to_num(self.tensor.sqrt(), 0)
        # return MCTensor(self.size(), nc=self.size_nc(-1),
        #                     val=value, device=self.device)

    def reshape(self, *new_shape):
        tmp = []
        nc = self.size_nc(-1)
        for i in range(nc):
            tmp.append(self.tensor[..., i].reshape(*new_shape))
        tmp = torch.stack(tmp, dim=-1)
        return MCTensor(tmp.size()[:-1], nc=nc, val=tmp, dtype=self.dtype, device=self.device)

    def squeeze(self, *args, **kwargs):
        return torch.squeeze(self, *args, **kwargs)

    def unsqueeze(self, *args, **kwargs):
        return torch.unsqueeze(self, *args, **kwargs)

    def view(self, *args, **kwargs):
        return self.tensor[..., 0].view(*args, **kwargs)

    def argmax(self, *args, **kwargs):
        return self.tensor.sum(-1).argmax(*args, **kwargs)

    def MCF_ok(self):
        """
        x: MCTensor to be checked check the the MCTensor is indeed a MCF 
        """
        for i in range(self.size_nc(-1)-1):
            assert torch.all(
                torch.abs(self.tensor[..., i]) >= torch.abs(self.tensor[..., i+1]))

    def to_sparse(self):
        if not self.is_sparse_grad:
            self.tensor = self.tensor.to_sparse()
            self.fc = self.fc.to_sparse()

    def to_dense(self):
        if self.is_sparse_grad:
            self.tensor = self.tensor.to_dense()
            self.fc = self.fc.to_dense()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, MCTensor))
            for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


@implements(torch.mean)
def mean(input):
    # this needs to be reimplemented
    # make sure this doesnt destroy its behaviors for standard tensors
    return input.tensor.sum(-1).mean()


def ensure_tensor(data):
    if isinstance(data, MCTensor):
        return data.tensor
    return torch.as_tensor(data)


@implements(torch.norm)
def norm(input, **kwargs):
    """
    this is an approximate version of norm for MCTensor, only use the first component to compute
    """

    # return torch.sqrt(((input*input).tensor[..., 0].sum()))
    return torch.norm(input.tensor.sum(-1), **kwargs)


@implements(torch.nn.functional.mse_loss)
def mse_loss(input, target, reduction="mean", size_average=None, reduce=None):
    assert input.size() == target.size(), f"input and target size are different!"
    to_return = (torch.norm(input - target))**2
    if reduction == "mean":
        return to_return/target.size().numel()
    elif reduction == "sum":
        return to_return


@implements(torch.narrow)
#  The returned MCTensor and input MCTensor share the same underlying storage.
def narrow(input, dim, start, length):
    return input.narrow(dim, start, length)


@implements(torch.clamp)
#  Not inplace, inplace should use clamp_
def clamp(input, min=None, max=None, *, out=None):
    return input.clamp(min=min, max=max)


@implements(torch.randn_like)
def randn_like(input, requires_grad=False, device=None, dtype=None):
    # only randn_like for the first component, double check device and cuda
    if device == None:
        device = input.device
    if dtype == None:
        dtype = input.dtype
    return MCTensor(input.size(), nc=input.size_nc(-1),
                    val=torch.randn(input.size()),
                    requires_grad=requires_grad, device=device, dtype=dtype)


@implements(torch.rand_like)
def rand_like(input, requires_grad=False, device=None, dtype=None):
    # only rand_like for the first component, double check device and cuda
    if device == None:
        device = input.device
    if dtype == None:
        dtype = input.dtype
    return MCTensor(input.size(), nc=input.size_nc(-1),
                    val=torch.rand(input.size()),
                    requires_grad=requires_grad, device=device, dtype=dtype)


@implements(torch.zeros_like)
def zeros_like(input, requires_grad=False, device=None, dtype=None):
    # only zeros_like for the first component, double check device and cuda
    if device == None:
        device = input.device
    if dtype == None:
        dtype = input.dtype
    return MCTensor(input.size(), nc=input.size_nc(-1),
                    val=torch.zeros(input.size()),
                    requires_grad=requires_grad, device=device, dtype=dtype)


@implements(torch.ones_like)
def ones_like(input, requires_grad=False, device=None, dtype=None):
    # only ones_like for the first component, double check device and cuda
    if device == None:
        device = input.device
    if dtype == None:
        dtype = input.dtype
    return MCTensor(input.size(), nc=input.size_nc(-1),
                    val=torch.ones(input.size()),
                    requires_grad=requires_grad, device=device, dtype=dtype)

###############################################################################


def AddMCN(x, y):
    """
    requires: x and y are both MCTensor,
              with same number of components and size
    """
    result = _AddMCN(x.tensor, y.tensor)
    return MCTensor(result.size()[:-1], nc=x.size_nc(-1),
                    val=result, device=x.device, dtype=x.dtype)


def Grow_ExpN(x, value):
    """
    requires: `x` is a MCTensor and `value` is a torch tensor.
              The dtype in `x` is equal to the dtype in `value`
    """
    if value.is_sparse:
        indices = tuple(value._indices())
        result = x.tensor.data
        result[indices] = _Grow_ExpN(result[indices], value._values())
    else:
        result = _Grow_ExpN(x.tensor, value)
    return MCTensor(result.size()[:-1], nc=x.size_nc(-1), val=result,
                    device=x.device, dtype=x.dtype)


@implements(torch.add)
def add(input, other, alpha=1):
    if isinstance(input, int) or isinstance(input, float):
        input = torch.tensor(input).to(other.tensor).unsqueeze(0)
    if isinstance(other, int) or isinstance(other, float):
        other = torch.tensor(other).to(input.tensor).unsqueeze(0)
    if alpha != 1:
        other = alpha * other
    if isinstance(input, MCTensor) and isinstance(other, MCTensor):  # Two MCTensors
        return AddMCN(input, other)
    elif isinstance(input, MCTensor):
        x = input  # the MCTensor
        y = other
    else:
        x = other  # the MCTensor
        y = input
    return Grow_ExpN(x, y)

###############################################################################


def MultMCN(x, y, slower=False):
    result = _MultMCN(x.tensor, y.tensor)
    return MCTensor(result.size()[:-1], nc=x.size_nc(-1),
                    val=result, device=x.device, dtype=x.dtype)


def ScalingN(x, value):
    stacked_h = _ScalingN(x.tensor, value)
    return MCTensor(stacked_h.size()[:-1], nc=x.size_nc(-1),
                    val=stacked_h, device=x.device, dtype=x.dtype)


@implements(torch.mul)
def mul(input, other):
    if isinstance(input, int) or isinstance(input, float):
        input = torch.tensor(float(input)).to(other.tensor).unsqueeze(0)
    if isinstance(other, int) or isinstance(other, float):
        other = torch.tensor(float(other)).to(input.tensor).unsqueeze(0)
    if (isinstance(input, MCTensor) and isinstance(
            other, MCTensor)):
        return MultMCN(input, other)
    elif isinstance(input, MCTensor):
        x = input  # the MCTensor
        y = other
    else:
        x = other  # the MCTensor
        y = input
    return ScalingN(x, y)

###############################################################################


def DivMC(x, y):
    if isinstance(x, MCTensor) and isinstance(y, torch.Tensor):
        inv = 1 / y
        result = _ScalingN(x.tensor, inv)
        return MCTensor(result.size()[:-1], nc=x.size_nc(-1),
                        val=result, device=x.device, dtype=x.dtype)
    elif isinstance(y, MCTensor) and isinstance(x, torch.Tensor):
        nc = y.tensor.size(-1)
        x_tensor = torch.zeros(x.size() + (nc,)).to(x)
        x_tensor[..., 0] = x
        result = _DivMCN(x_tensor, y.tensor)
        return MCTensor(result.size()[:-1], nc=y.size_nc(-1),
                        val=result, device=y.device, dtype=y.dtype)
    elif isinstance(x, MCTensor) and isinstance(y, MCTensor):
        result = _DivMCN(x.tensor, y.tensor)
        return MCTensor(result.size()[:-1], nc=x.size_nc(-1),
                        val=result, device=x.device, dtype=x.dtype)
    else:
        raise NotImplementedError()


@implements(torch.div)
def div(x, y):
    if isinstance(x, int) or isinstance(x, float):
        x = torch.tensor(float(x)).to(y.tensor).unsqueeze(0)
    if isinstance(y, int) or isinstance(y, float):
        y = torch.tensor(float(y)).to(x.tensor).unsqueeze(0)
    return DivMC(x, y)

###############################################################################


def Dot_MCN(x, y):
    # produce a sclar
    assert x.dim() == 1 and y.dim() == 1 and x.size(
        0) == y.size(0), f"Check data before input"
    assert isinstance(x, MCTensor) and isinstance(y, torch.Tensor)
    return _Dot_MCN(x.tensor, y)


@implements(torch.dot)
def dot(input, other):
    if isinstance(input, MCTensor) and isinstance(other, torch.Tensor):
        x = input
        y = other
    elif isinstance(input, torch.Tensor) and isinstance(other, MCTensor):
        x = other
        y = input
    else:
        raise NotImplemented
    return Dot_MCN(x, y)

###############################################################################


def MV_MCN(x, y):
    assert x.dim() == 2 and y.dim() == 1 and x.size(
        1) == y.size(0), f"Check data before input"
    if (isinstance(x, MCTensor) and isinstance(y, torch.Tensor)):
        result = _MV_MC_T(x.tensor, y)
        return MCTensor(result.size()[:-1], nc=x.size_nc(-1), val=result)
    elif (isinstance(x, torch.Tensor) and isinstance(y, MCTensor)):
        result = _MV_T_MC(x, y.tensor)
        return MCTensor(result.size()[:-1], nc=y.size_nc(-1), val=result)
    else:
        raise NotImplemented


@implements(torch.mv)
def mv(input, other):
    if input.dim() == 2 and other.dim() == 1:
        x = input  # matrix
        y = other  # vector
    elif input.dim() == 1 and other.dim() == 2:
        x = other  # matrix
        y = input  # vector
    else:
        raise NotImplemented
    return MV_MCN(x, y)

###############################################################################


def MM_MCN(x, y):
    assert x.dim() == 2 and y.dim() == 2 and x.size(
        1) == y.size(0), f"Check data before input"
    if (isinstance(x, MCTensor) and isinstance(y, torch.Tensor)):
        tmp = _MM_MC_T(x.tensor, y)
        result = MCTensor(x.tensor.size(0), y.size(1), nc=x.size_nc(-1),
                          val=tmp)
    elif (isinstance(x, torch.Tensor) and isinstance(y, MCTensor)):
        tmp = _MM_T_MC(x, y.tensor)
        result = MCTensor(x.size(0), y.size(1), nc=y.size_nc(-1),
                          val=tmp)
    else:
        raise NotImplemented
    return result


@implements(torch.mm)
def mm(input, other):
    return MM_MCN(input, other)
###############################################################################


def BMM_MCN(x, y):
    assert (x.dim() == 3 and y.dim() == 3), "expected dim 3 input in both side"
    if isinstance(x, MCTensor) and isinstance(y, torch.Tensor):
        tmp, size, nc = _BMM_MC_T(x.tensor, y)
        return MCTensor(size, nc=nc, val=tmp)
    elif isinstance(x, torch.Tensor) and isinstance(y, MCTensor):
        tmp, size, nc = _BMM_T_MC(x, y.tensor)
        return MCTensor(size, nc=nc, val=tmp)
    raise NotImplemented


@implements(torch.bmm)
def bmm(input, other):
    return BMM_MCN(input, other)
###############################################################################


def Matmul_MCN(x, y):
    assert x.device == y.device, f"inputs must be on the same device"
    assert x.dtype == y.dtype, f"inputs must be same dtype"
    x_dim, y_dim = x.dim(), y.dim()
    if x_dim == 1 and y_dim == 1:
        return torch.dot(x, y)
    elif x_dim == 2 and y_dim == 2:
        return torch.mm(x, y)
    elif (x_dim == 2 and y_dim == 1) or (x_dim == 1 and y_dim == 2):
        return torch.mv(x, y)
    elif (x_dim > 2 and y_dim == 1) or (x_dim == 1 and y_dim > 2):
        return torch.mul(x, y)
    elif x_dim == y_dim and x_dim == 3:
        if isinstance(x, MCTensor):
            tmp, size, nc = _BMM_MC_T(x.tensor, y)
        else:
            tmp, size, nc = _BMM_T_MC(x, y.tensor)
        return MCTensor(size, nc=nc, val=tmp)
    elif x_dim == y_dim and x_dim == 4:
        if isinstance(x, MCTensor):
            tmp, size, nc = _4DMM_MC_T(x.tensor, y)
        else:
            tmp, size, nc = _4DMM_T_MC(x, y.tensor)
        return MCTensor(size, nc=nc, val=tmp)
    elif x_dim > y_dim:
        y = y[(None,) * (x_dim - y_dim)]  # unsqueeze
        if x_dim == 3:
            if isinstance(x, MCTensor):
                tmp, size, nc = _BMM_MC_T(x.tensor, y)
            else:
                tmp, size, nc = _BMM_T_MC(x, y.tensor)
            return MCTensor(size, nc=nc, val=tmp)
        elif x_dim == 4:
            if isinstance(x, MCTensor):
                tmp, size, nc = _4DMM_MC_T(x.tensor, y)
            else:
                tmp, size, nc = _4DMM_T_MC(x, y.tensor)
            return MCTensor(size, nc=nc, val=tmp)
    elif x_dim < y_dim:
        x = x[(None,) * (y_dim - x_dim)]  # unsqueeze
        if y_dim == 3:
            if isinstance(x, MCTensor):
                tmp, size, nc = _BMM_MC_T(x.tensor, y)
            else:
                tmp, size, nc = _BMM_T_MC(x, y.tensor)
            return MCTensor(size, nc=nc, val=tmp)
        elif y_dim == 4:
            if isinstance(x, MCTensor):
                tmp, size, nc = _4DMM_MC_T(x.tensor, y)
            else:
                tmp, size, nc = _4DMM_T_MC(x, y.tensor)
            return MCTensor(size, nc=nc, val=tmp)
    raise NotImplemented


@implements(torch.matmul)
def matmul(input, other):
    return Matmul_MCN(input, other)
###############################################################################


@implements(torch.addmm)
def addmm(input, mat1, mat2, beta=1.0, alpha=1.0):
    return beta * input + alpha * (mat1 @ mat2)


@implements(torch.squeeze)
def squeeze(input, *args, **kwargs):
    if isinstance(input, torch.Tensor):
        return input.squeeze(*args, **kwargs)
    elif isinstance(input, MCTensor):
        tensor_rep = input.tensor.squeeze(*args, **kwargs)
        return MCTensor(tensor_rep.size()[:-1], nc=input.size_nc(-1), val=tensor_rep,
                        device=input.device, dtype=input.dtype)
    raise NotImplemented


@implements(torch.unsqueeze)
def unsqueeze(input, *args, **kwargs):
    if isinstance(input, torch.Tensor):
        return input.unsqueeze(*args, **kwargs)
    elif isinstance(input, MCTensor):
        tensor_rep = input.tensor.unsqueeze(*args, **kwargs)
        return MCTensor(tensor_rep.size()[:-1], nc=input.size_nc(-1), val=tensor_rep,
                        device=input.device, dtype=input.dtype)
    raise NotImplemented


@implements(torch.transpose)
def transpose(x, *args, **kw):
    T = []
    nc = x.size_nc(-1)
    for i in range(nc):
        T.append(x.tensor[..., i].transpose(*args, **kw))
    T = torch.stack(T, dim=-1)
    return MCTensor(T.size()[:-1], nc=nc, val=T, device=x.device, dtype=x.dtype)


################## implements torch.nn.functional function ##################
@implements(torch.nn.functional.relu)
def relu(input, inplace=False):
    fc = torch.nn.functional.relu(input.fc, inplace=inplace)
    if inplace == False:
        input = input.clone()
    input.fc = fc
    input.tensor[..., :1] = input.fc
    # setting all components base on presence of zeros in fc
    input.tensor[..., 1:] = torch.where(
        fc > 0, input.tensor[..., 1:], torch.zeros_like(fc))
    return input


@implements(torch.sigmoid)
def sigmoid(input):
    return 1/(1+torch.exp(-input))

# @implements(torch.nn.functional.layer_norm)
# def layer_norm(input):
#     size, nc = input.size()[:-1], input.size_nc()[-1]
#     ret = torch.nn.functional.layer_norm(input.tensor.sum(-1))
#     return MCTensor(size, nc=nc, val=ret, device=input.device, dtype=input.dtype)

@implements(torch.nn.functional.softmax)
# warning: to match the performance of standard Tensor with these parameters
def softmax(input, dim, _stacklevel=3):
    device = input.device
    dtype = input.dtype
    size, nc = input.size(), input.size_nc(-1)
    exp_input = torch.exp(input)
    exp_input_tensor = exp_input.tensor
    if dim is None:
        dim = -1
    if dim < 0:
        dim += len(size)
    denom_size = size[:dim] + (1,) + size[dim+1:]
    # print("denom init size:",denom_size)
    denom_shape = denom_size + (nc,)
    numerator = exp_input
    denominator = MCTensor(denom_size, nc=nc,
                           val=exp_input_tensor.select(
                               dim, 0).reshape(denom_shape),
                           device=device, dtype=dtype)
    for i in range(1, size[dim]):
        tmp = MCTensor(denom_size, nc=nc,
                       val=exp_input_tensor.select(
                           dim, i).reshape(denom_shape),
                       device=device, dtype=dtype)
        denominator = tmp + denominator
    res = numerator / denominator
    return res.tensor[..., 0]


@implements(torch.cat)
def cat(mctensors, **kw):
    single_MC = mctensors[0]
    dtype = single_MC.dtype
    device = single_MC.device
    nc = single_MC.size_nc(-1)
    values = torch.cat([mc.tensor for mc in mctensors], **kw)
    return MCTensor(values.size(), nc=nc, val=values, device=device, dtype=dtype)


@implements(torch.stack)
def stack(mctensors, **kw):
    single_MC = mctensors[0]
    dtype = single_MC.dtype
    device = single_MC.device
    nc = single_MC.size_nc(-1)
    values = torch.stack([mc.tensor for mc in mctensors], **kw)
    return MCTensor(values.size()[:-1], nc=nc, val=values, device=device, dtype=dtype)


@implements(torch.pow)
def pow(input, exponent):
    tmp = _pow(input.tensor, exponent)
    return MCTensor(input.size(), nc=input.size_nc(-1), val=tmp)


@implements(torch.isnan)
def isnan(input):
    return torch.isnan(input.data)


@implements(torch.exp)
def exp(input):
    tmp = _exp(input.tensor)
    return MCTensor(input.size(), nc=input.size_nc(-1), val=tmp)


@implements(torch.erf)
def erf(input):
    tmp = torch.erf(input.tensor)
    return MCTensor(input.size(), nc=input.size_nc(-1), val=tmp)


@implements(torch.tanh)
def tanh(input):
    tmp = torch.tanh(input.tensor)
    return MCTensor(input.size(), nc=input.size_nc(-1), val=tmp)


@implements(torch.nn.functional.dropout)
def dropout(input, p=0.5, training=True, inplace=False):
    if training:
        fc_tensor = torch.nn.functional.dropout(input.tensor.sum(-1), p=p)
        if inplace:
            input.tensor[..., 0].data.copy_(fc_tensor)
            if input.size_nc(-1) > 1:
                input.tensor[..., 1:].data.zero_()
            return input
        else:
            ret_MCTensor_like = torch.zeros_like(input.tensor)
            ret_MCTensor_like[..., 0] = fc_tensor
            return MCTensor(ret_MCTensor_like.size()[:-1], nc=input.size_nc(-1), \
                val=ret_MCTensor_like, device=input.device, dtype=input.dtype)
    else:
        return input


@implements(torch.log)
def log(input):  # approximat via first dim
    tmp = torch.zeros_like(input.tensor)
    tmp[..., 0] = torch.log(input.tensor[..., 0])
    return MCTensor(input.size(), nc=input.size_nc(-1),
                    val=tmp, device=input.device, dtype=input.dtype)


@implements(torch.square)
def square(input):
    tmp = _square(input.tensor)
    return MCTensor(input.size(), nc=input.size_nc(-1),
                    val=tmp, device=input.device, dtype=input.dtype)


@implements(torch.nn.functional.linear)
def linear(input, weight, bias=None):
    ret = torch.matmul(input, weight.T)
    if bias is None:
        return ret
    else:
        return ret + bias
    # if input.dim() == 2:
    #     x1, x2 = input.size()
    #     y1, y2 = weight.size()
    #     if x2 == y2:
    #         tmp = []
    #         for i in range(x1):
    #             tmp.append(torch.mv(weight, input[i, :]))
    #         ret = torch.stack(tmp)
    #     else:
    #         ret = mm(input, weight)
    #     if bias is not None:
    #         return ret + bias.to(device)
    #     else:
    #         return ret
    # elif input.dim() == 1:
    #     ret = torch.mv(weight, input)
    #     if bias is not None:
    #         return ret + bias.to(device)
    #     else:
    #         return ret
    # else:
    #     ret = torch.matmul(input, weight)
    #     if bias is not None:
    #         return ret + bias.to(device)
    #     else:
    #         return ret


@implements(torch.diag)
def diag(input):
    assert input.dim() == 2 or input.dim() == 1
    dtype = input.dtype
    device = input.device
    nc = input.size_nc(-1)
    if input.dim() == 1:
        n = input.size(0)
        ret = torch.zeros(n, n, nc, dtype=dtype, device=device)
        for i in range(n):
            ret[i, i] = input.tensor[i]
        return MCTensor(ret.size()[:-1], nc=nc, val=ret, device=device, dtype=dtype)
    else:
        n = input.size(0)
        ret = torch.zeros(n, nc, dtype=dtype, device=device)
        for i in range(n):
            ret[i, :] = input.tensor[i, i]
        return MCTensor(ret.size()[:-1], nc=nc, val=ret, device=device, dtype=dtype)


@implements(torch.nn.functional.nll_loss)
def nll_loss(input, target):
    return torch.mean(torch.diag(-input[:, target]))


@implements(torch.nn.functional.log_softmax)
def log_softmax(input, dim, _stacklevel=3, dtype=None):
    return torch.log(torch.nn.functional.softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype))
