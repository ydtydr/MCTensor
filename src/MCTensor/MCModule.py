from .MCTensor import MCTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

nn.Module

class MCModule(object):

    def __init__(self):
        self.training = True

    def register_parameter(self, name, param):
        if param is None:
            pass
        elif param.grad_fn:
            raise ValueError("Nonleaf parameter is not allowed!")
        else:
            pass  # ignore
    
    def train(self, mode=True):
        self.training = mode
        for module in self.parameters():
            if isinstance(module, MCModule) or isinstance(module, nn.Module):
                module.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
        
    def parameters(self):
        for _, param in self.named_parameters():
            yield param

    def named_parameters(self, prefix=''):
        for name, param in self.__dict__.items():
            new_name = name if prefix == '' else prefix + '.' + name
            if isinstance(param, MCModule) or isinstance(param, nn.Module):
                yield from param.named_parameters(prefix=new_name)
            elif isinstance(param, MCTensor) and param.requires_grad:
                yield (new_name, param)
            # elif isinstance(param, MCSequential):
            #     for i, p in enumerate(param.layers):
            #         if isinstance(p, MCModule) or isinstance(p, nn.Module):
            #             yield from p.named_parameters(prefix=new_name)
            #         elif isinstance(p, MCTensor) and p.requires_grad:
            #             yield (new_name, param)

    def forward(self, *args, **kwds):
        raise NotImplementedError("Base class!")

    def to_device_(self, device):
        for name, param in self.__dict__.items():

            if isinstance(param, MCModule):
                self.__dict__[name] = param.to_device_(device)
                # setattr(self, name, param.to_device_(device))
                # param.to_device_(device)
            if isinstance(param, MCTensor):
                self.__dict__[name] = param.to(device)
                # setattr(self, name, param.to(device))
            if isinstance(param, nn.Module):
                self.__dict__[name] = param.to(device)
                # setattr(self, name, param.to(device=device))
                
        return self

    def to_type_(self, dtype):
        for name, param in self.__dict__.items():
            if isinstance(param, MCModule):
                param.to_type_(dtype)
            if isinstance(param, MCTensor):
                setattr(self, name, param.to(dtype))
            if isinstance(param, nn.Module):
                setattr(self, name, param.to(dtype=dtype))
        return self

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def load_weight(self, another_model:nn.Module):
        other_params_generator = another_model.named_parameters()
        for p in self.parameters():
            _, other_p = next(other_params_generator)
            if isinstance(p, MCTensor) and isinstance(other_p, torch.Tensor):
                p.tensor[..., 0].data.copy_(other_p.data)
                if p.size_nc(-1) > 1:
                    p.tensor[..., 1:].data.zero_()
                p.fc.grad = None
            elif isinstance(p, torch.Tensor) and isinstance(other_p, torch.Tensor):
                p.data.copy_(other_p.data)
                p.grad = None
            elif isinstance(p, MCTensor) and isinstance(other_p, MCTensor):
                p.tensor.data.copy_(other_p.tensor)
                p.fc.grad = None
            elif isinstance(p, torch.Tensor) and isinstance(other_p, MCTensor):
                p.data.copy_(other_p.tensor[..., 0].data)
                p.grad = None
            else:
                raise NotImplementedError()
        return self
    
    def apply(self, method):
        for name, param in self.__dict__.items():
            if isinstance(param, MCModule):
                param.apply(method)
            elif isinstance(param, nn.Module):
                param.apply(method)
            elif isinstance(param, MCTensor) and param.requires_grad:
                param = method(param)
        return self

    def cuda_(self):
        return self.to_device_(torch.device('cuda'))

    def cpu_(self):
        return self.to_device_(torch.device('cpu'))

    def fp16_(self):
        return self.to_(torch.float16)


class MCSequential(MCModule):
    def __init__(self, *args):
        super(MCSequential, self).__init__()
        self.layers = args
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def named_parameters(self, prefix=''):
        for i, p in enumerate(self.layers):
            if isinstance(p, MCModule) or isinstance(p, nn.Module):
                yield from p.named_parameters(prefix=prefix)
            elif isinstance(p, MCTensor) and p.requires_grad:
                yield (prefix, p)

class MCModuleList(MCModule):
    def __init__(self, lst):
        super(MCModuleList, self).__init__()
        if isinstance(lst, list):
            self.layer = lst
        else:
            self.layer = list(lst)
    
    def __getitem__(self, idx):
        return self.layer[idx]

    def named_parameters(self, prefix=''):
        for i, p in enumerate(self.layer):
            new_name = prefix + "." + str(i)
            if isinstance(p, MCModule) or isinstance(p, nn.Module):
                yield from p.named_parameters(prefix=new_name)
            elif isinstance(p, MCTensor) and p.requires_grad:
                yield (new_name, p)

class MCLinear(MCModule):
    def __init__(self, in_features, out_features, nc, bias=True,
                 device=None, dtype=None, _weight=None):
        super(MCLinear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.nc = nc
        if _weight is None:
            self.weight = MCTensor(out_features, in_features,
                                   nc=nc, requires_grad=True,
                                   **factory_kwargs)
        else:
            self.weight = MCTensor(out_features, in_features,
                                   nc=nc, requires_grad=True,
                                   val=_weight,
                                   **factory_kwargs)

        if bias:
            self.bias = MCTensor(out_features,
                                 nc=nc, requires_grad=True,
                                 **factory_kwargs)
        else:
            self.bias = None

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, nc = {}'.format(
            self.in_features, self.out_features, self.bias is not None, self.nc,
        )


class MCEmbedding(MCModule):
    def __init__(self, num_embeddings, embedding_dim, nc,
                 padding_idx=None, max_norm=None, norm_type=2, sparse=False,
                 device=None, dtype=None, _weight=None):
        super(MCEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.nc = nc
        self.sparse = sparse
        if _weight is None:
            self.weight = MCTensor(num_embeddings, embedding_dim, nc=nc,
                                   requires_grad=True, device=device, sparse_grad=sparse, dtype=dtype)
        else:
            self.weight = MCTensor(num_embeddings, embedding_dim, nc=nc, val=_weight,
                                   requires_grad=True, device=device, sparse_grad=sparse, dtype=dtype)

    def init_weights(self, weight, scale=1e-4):
        raise NotImplementedError()

    def forward(self, inputs):
        return self.weight[inputs]

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)
