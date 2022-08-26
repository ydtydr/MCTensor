# MCTensor: A High-Precision Deep Learning Library with Multi-Component Floating-Point

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache-2.svg)](https://opensource.org/licenses/Apache-2.0)

*MCTensor* is a general-purpose, fast, and high-precision deep learning library built upon pytorch and compliant on PyTorch programming paradigm on building deep learning modules and training codes. *MCTensor* follows the multiple-component floating-point format (MCF) using an unevaluated sum of multiple ordinary floating-point numbers (e.g. float16, float32, float64) to represent a high-precision floating point number. 

The paper is presented in *Hardware Aware Efficient Training* workshop (HAET) in ICML'22 and its pdf version can be found in https://arxiv.org/pdf/2207.08867.pdf. 

# Install
## Requirements
Python >= 3.6
PyTorch >= 1.10.0
CUDA >= 10.1 on linux

## Procedures
1. clone this repo locally
2. run `python build.py install`.


# Library Structure
The folder `src/MCTensor` contains source code for the library, specifically, 
- `MCTensor.py` stores the `MCTensor` class definition and encapsulated high-level operators. 
- `MCOpBasic.py` stores the MCF algorithms such as `Two_Sum`, `Renormalize`, and `Two_Prod` and middle-level callers as `_AddMCN`, `_MultMCN`, and `_DivMCN` operators. They will be called from `MCTensor`
- `MCOpMatrix.py` stores the MCF algorithms for vector and matrix level operators such as `_Dot_MCN` , `_MV_MC_T`, and `_MM_MC_T`.
- `MCOptim.py` stores the MC-optimizers used for training `MCModule` and native pytorch modules.
- `MCModule.py` stores basic `MCModule` definition such as `MCLinear` and `MCEmbedding`.

## MCTensor
An `MCTensor` `x` contains a few important attributes, namely, 
- **nc**: the number of components, which can be derived with `x.size_nc(-1)`.
- **tensor**, the underlying data with all components, whose last dimension is the component dimension, i.e., `x.tensor[...,i]` will retrive the `i`-th component.
- **fc** is a view of the first component, retrived with `x.fc`, which is used mainly for tracking gradient graph so as to be consistent with PyTorch autograd mechanism. 

## Operators and Sample Codes
Basic operators are implemented first for **MCTensor** from basic MCF algorithms described in the paper (e.g., `Two_Sum`, `_Simple_renormalize`). For example, `add`, `sub`, `div` and `mul`. Matrix operators are then developed, including common ones adopted in PyTorch with same semantics, e.g., `dot`, `mv`, `mm`, `bmm`, `matmul`, except for `matmul` where we only support at most 4-d tensors matmul at this moment. 

We provide some sample codes for better illustration. `MCTensor` overrides PyTorch operators for `MCTensor`-`MCTensor` , and `MCTensor` -`Tensor` arithmetic with decorator hooks. Such hook works for `torch.FUNC` or `mc_tensor.FUNC` calls. 

- `torch.add(mc_tensor, tensor)` works
- `torch.add(tensor, mc_tensor)` works
- `mc_tensor + tensor` works
- `mc_tensor.add(tensor)` works
- `tensor + mc_tensor` works

The following are some sample MCTensor definition and arithmetics codes.
```python
>>> MC_A  =  MCTensor((2, 2), nc=2)
>>> MC_A
MCTensor(Size=torch.Size([2, 2]), number of components=2, requires_grad=False)
>>> MC_A.tensor[..., 0]
tensor([[-1.8906, 0.3968], 
        [ 0.8522, -1.0379]])
>>> (MC_A + MC_A).fc
tensor([[[-3.7812], 
         [ 0.7937]], 
        [[ 1.7044], 
         [-2.0758]]])
>>> B = torch.ones(2, 2)
>>> MC_B = MCTensor((2, 2), val=B, nc=2)
>>> MC_C = MC_A  +  MC_B
>>> MC_C.fc
tensor([[[-0.8906], 
         [ 1.3968]], 
        [[ 1.8522], 
         [-0.0379]]])
>>> MC_A_cuda = MC_A.cuda()
>>> MC_AB = MC_A * MC_B
tensor([[[-1.8906], 
         [ 0.3968]], 
        [[ 0.8522], 
         [-1.0379]]])
>>> MC_A[0]
MCTensor(Size=torch.Size([2]), number of components=2, requires_grad=False)
>>> torch.dot(MC_A[0], B[0])
tensor(-1.4938)
>>> MC_A_requires_grad = MCTensor(2, 2, nc=2, val=MC_A, requires_grad=True)
>>> MC_A_requires_grad.sum().backward()
>>> MC_A_requires_grad.grad
tensor([[1., 1.], 
        [1., 1.]])

```



## MCModule
`MCModule` is the basic MC Module definition block, similar to `nn.Module`. `MCModule` uses `MCTensor` whose `requires_grad=True` for trainable parameters, which means **ANY fields in MCModule with `requires_grad=True` will be passed as a parameter to optimizer with `mc_module.parameters()` call**. Currently, `buffer` is not supported in `MCModule`.

Some example layers including 
- `MCLinear`, inherited from `MCModule` class, follows the implementation of `nn.Linear` in PyTorch.
- `MCEmbedding`, inherited from `MCModule` class, follows the implementation of `nn.Embedding` in PyTorch.
- `MCSequential`, `MCModuleList` ... 

Just as the PyTorch case, users can develop advanced models with `MCModule`, for example,  
```python
class MCMLP(MCModule):
    def __init__(self, input_dim, hidden1, hidden2, nc=2, dtype=d16, device=device):
        super(MCMLP, self).__init__()
        self.fc1 = MCLinear(input_dim, hidden1, nc=nc, bias=False, dtype=dtype, device=device)
        self.fc2 = MCLinear(hidden1, hidden2, nc=nc, bias=False, dtype=dtype, device=device)
        self.fc3 = MCLinear(hidden2, 1, nc=nc, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.tensor.sum(-1) # transform x to standard tensor for efficiency
        x = F.relu(self.fc2(x))
        x = x.tensor.sum(-1)
        x = self.fc3(x)
        x = torch.sigmoid(x)        
        x = x.tensor.sum(-1)
        return x 
```

## MCOptimizer
`MCSGD` and `MCAdam` are implemented as counterparts of PyTorch `SGD` and `Adam` optimizers, which both inherit from `MCOptimizer` class. Just as the usage of optimizers in PyTorch, `MCOptimizer` can be used in the same way during training as 
```python
model = MCMLP(input_dim, hidden1, hidden2, nc=nc, 
              device=torch.device('cuda:0'), dtype=torch.float16)
criterion  =  torch.nn.BCELoss()
mc_optimizer = MCOptim.MCSGD(model.parameters(), lr)

for X, Y in trainloader:
    mc_optimizer.zero_grad()
    X, Y = X.cuda(), Y.cuda()
    Y_hat = model(X)
    loss = criterion(Y_hat, Y)
    mc_outputs.backward()
    mc_optimizer.step()
```

## Applications
We provide codes for experiments in the paper in the `applications` folder, including `basic_examples` and `poincare_embedding`. 

# Extending MCTensor
MCTensor uses `implements` decorator as defined in `MCTensor.py` to override PyTorch operators. For example, we can override PyTorch's `torch.cat` as

```
@implements(torch.cat)
def cat(mctensors, *args, **kw):
	print("this is my mctensor cat")
    
>>> cat(MCTensor(1, 2, nc=2))
this is my MCTensor cat
``` 


# Authors

 - [Tao Yu](https://www.cs.cornell.edu/~tyu/), tyu@cs.cornell.edu 
 - Wentao Guo, wg247@cornell.edu
 - Jianan Canal Li, jl3789@cornell.edu 
 - Tiancheng Yuan, ty373@cornell.edu 
 - [Christopher De Sa](https://www.cs.cornell.edu/~cdesa/), cdesa@cs.cornell.edu


# Acknowledgement and disclaimer
This work is supported by NSF IIS-2008102. *MCTensor* endeavors to follow the semantics and speed of PyTorch. As it is still under development and mainly implemented in Python level, it may not achieve the same speed as native PyTorch did, and sometimes their semantics are not fully equivalent. Please perform a correctness and performance test before the deployment and feel free to leave a issue or contact us.


# License
MCTensor uses Apache-2 license in the [LICENSE](https://github.com/ydtydr/Hyperbolic_Library/tree/release/LICENSE) file.


# Cite us

If you find MCTensor library helpful in your research, please consider citing us:

    @misc{https://doi.org/10.48550/arxiv.2207.08867,
      doi = {10.48550/ARXIV.2207.08867},
      url = {https://arxiv.org/abs/2207.08867},
      author = {Yu, Tao and Guo, Wentao and Li, Jianan Canal and Yuan, Tiancheng and De Sa, Christopher},
      title = {MCTensor: A High-Precision Deep Learning Library with Multi-Component Floating-Point},
      publisher = {arXiv},
      year = {2022},
      copyright = {Creative Commons Attribution 4.0 International}
    }
