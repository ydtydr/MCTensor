#!/usr/bin/env python3
from MCTensor.MCOpBasics import _AddMCN, _ScalingN, _DivMCN
from MCTensor import MCTensor
import sys
import os
from turtle import st
import torch as th
from torch.autograd import Function
from zmq import device
from .common import acosh
from .manifold import Manifold
sys.path.append(os.path.abspath('../../../'))


class MC_HalfspaceManifold(Manifold):
    __slots__ = ["eps", "_eps", "norm_clip", "max_norm", "debug"]

    @staticmethod
    def dim(dim):
        return dim

    def __init__(self, eps=1e-12, _eps=1e-5, norm_clip=1, max_norm=1e6,
                 debug=False, **kwargs):
        self.eps = eps
        self._eps = _eps
        self.norm_clip = norm_clip
        self.max_norm = max_norm
        self.debug = debug

    def sinhc(self, u):
        return th.div(th.sinh(u), u)

    def to_poincare_ball(self, u):
        u = u.tensor.sum(-1)
        d = u.size(-1)
        uu = th.zeros(u.size(0), d + 1).to(u.device)
        squnom = th.sum(th.pow(u, 2), dim=-1)  # n
        uu[..., 0] = th.div(th.ones_like(u[..., -1]), u[..., -1]
                            ) + th.div(squnom, 4 * u[..., -1])  # n
        uu[..., 1] = th.div(th.ones_like(u[..., -1]), u[..., -1]
                            ) - th.div(squnom, 4 * u[..., -1])  # n
        uu[..., 2:] = th.div(
            u[..., :d - 1], u[..., -1].unsqueeze(-1).expand_as(u[..., :d - 1]))
        return uu.narrow(-1, 1, d) / (uu.narrow(-1, 0, 1) + 1)

    def distance(self, uu, vv):
#         dis = MC_HalfspaceDistance(uu, vv)
        dis = MC_HalfspaceDistance.apply(uu.tensor, vv.tensor)
        return dis

    def pnorm(self, u):
        return th.sqrt(th.sum(th.pow(self.to_poincare_ball(u), 2), dim=-1))

    def normalize(self, w):
        """Normalize vector such that it is located on the halfspace model"""
#         d = w.size(-1)
        # narrow: take last dim
        narrowed = w.data.narrow(-2, -1, 1)
        narrowed0 = narrowed.narrow(-1, 0, 1)
        # clamp_: inplace make all entries greater than 0.0,
        # change to 1e-8 if any went wrong
        narrowed0.clamp_(min=0.0)
#         print(narrowed0.size(), narrowed.data[..., 1:].size(), th.zeros_like(narrowed0).size())
#         raise
        narrowed.data[..., 1:] = th.where(narrowed0 > 0, narrowed.data[..., 1:],
                                          th.zeros_like(narrowed0))
        return w

    def init_weights(self, w, irange=1e-5):
        th.manual_seed(40)
        device = w.tensor.device
        dtype = w.tensor.dtype
        w.data.zero_()
        w.fc.data.zero_()
        #########
        w.fc.data[..., :-1, 0].uniform_(-irange, irange)
        w.fc.data[..., -1, 0] = 1.0 + irange * 2 * (th.rand_like(w.fc[..., -1, 0], device=device, dtype=dtype) - 0.5) 
        w.tensor = th.cat(
            [w.fc, th.zeros(w.size() + (w.size_nc(-1)-1,), device=device, dtype=dtype)], dim=-1) 


    def rgrad(self, p, d_p):
        """Euclidean gradient for halfspace"""
        if d_p.is_sparse:
            u = d_p._values()
            x = p.index_select(0, d_p._indices().squeeze()).sum(-1)
        else:
            u = d_p
            x = p
        # transform from Euclidean grad to Riemannian grad
        # warning: following gradient can be MC gradient, now a tensor gradient
        u.mul_((x[..., -1]).unsqueeze(-1))
        return d_p
    
            
    def expm(self, p, d_p, lr=None, out=None, normalize=False):
        """Exponential map for halfspace model"""
        d = p.size(-2)
        if out is None:
            out = p
        if d_p.is_sparse:
            ix, d_val = d_p._indices().squeeze(), d_p._values()
            p_val = self.normalize(p.index_select(0, ix))
            newp_val = p_val.clone()
            # Numerical stable form of the exponential map
            mask_pos = d_val[..., -1] > 0
            mask_neg = d_val[..., -1] <= 0
            Pos = d_val[mask_pos]
            Neg = d_val[mask_neg]
            if len(Pos) != 0:
                s_postive = th.norm(Pos, dim=-1)
                r_square = th.sum(th.pow(Pos[..., :-1], 2), dim=-1)
                scoths = th.div(s_postive, th.tanh(s_postive))
                scschs_square = th.pow(
                    th.div(s_postive, th.sinh(s_postive)), 2)
                ##=======================================================================##
                # # One scalingN version of gradient update
                grad = th.div(
                    scoths + Pos[..., -1], scschs_square + r_square).unsqueeze(-1) * Pos[..., :-1]
                grad = _ScalingN(p_val[..., -1, :]
                                 [mask_pos], grad, style='EXPM')
                ##=======================================================================##
                # # Two scalingN version of gradient update
                # grad = th.div(scoths + Pos[..., -1], scschs_square + r_square)
                # grad = _ScalingN(p_val[..., -1, :][mask_pos], grad)
                # grad = _ScalingN(grad, Pos[..., :-1], style='EXPM')
                ##=======================================================================##
                # updating MCEmb weight for till last dim
                newp_val[..., :-1,
                         :][mask_pos] = _AddMCN(p_val[..., :-1, :][mask_pos], grad)
                grad = th.div(s_postive + Pos[..., -1] * th.tanh(s_postive), r_square +
                              th.pow(th.div(Pos[..., -1], th.cosh(s_postive)), 2)) * th.div(s_postive, th.cosh(s_postive))
                grad = _ScalingN(p_val[..., -1, :][mask_pos], grad)
                # updating MCEmb weight for last dim
                newp_val[..., -1, :][mask_pos] = grad
            if len(Neg) != 0:
                s_negative = th.norm(Neg, dim=-1)
                zeros_mask = (s_negative == 0.0)
                coshs = th.cosh(s_negative)
                sihncs = self.sinhc(s_negative)
                sihncs[zeros_mask] = th.ones_like(sihncs[th.isnan(sihncs)])
                # updating MCEmb weight for till last dim
                grad = th.zeros_like(p_val[..., -1, :][mask_neg])
                grad[..., 0] = th.div(coshs, sihncs)-Neg[..., -1]
                grad = _DivMCN(p_val[..., -1, :][mask_neg], grad)
                grad = _ScalingN(grad, Neg[..., :-1], style="EXPM")
                newp_val[..., :-1,
                         :][mask_neg] = _AddMCN(p_val[..., :-1, :][mask_neg], grad)
                # updating MCEmb weight for last dim
                grad = th.zeros_like(p_val[..., -1, :][mask_neg])
                grad[..., 0] = coshs-Neg[..., -1]*sihncs
                newp_val[..., -1,
                         :][mask_neg] = _DivMCN(p_val[..., -1, :][mask_neg], grad)
            ####################
            newp_val = self.normalize(newp_val)
            p.index_copy_(0, ix, newp_val)
        else:
            raise NotImplementedError

class MC_HalfspaceDistance(Function):
    @staticmethod
    def forward(self, u, v):
        u = MCTensor(u.size()[:-1], nc=u.size(-1), val=u)
        v = MCTensor(v.size()[:-1], nc=v.size(-1), val=v)
        assert th.isnan(u).max().sum().item() == 0, "u includes NaNs"
        assert th.isnan(v).max().sum().item() == 0, "v includes NaNs"
        if len(u) < len(v):
            u = u.expand_as(v)
        elif len(u) > len(v):
            v = v.expand_as(u)
        self.save_for_backward(u.tensor, v.tensor)
        d = u.size(-1)
        sq_diff = th.square(u - v).tensor.sum(-1)
        sqnorm = th.sum(sq_diff, dim=-1)
        self.x_ = th.div(sqnorm, 2.0 * u[..., -1] * v[..., -1])  # MCTensor
        self.z_ = th.sqrt((self.x_ * (self.x_ + 2.0)).tensor.sum(-1))  # tensor
#         assert th.isnan(self.z_).max() == 0, f"self.z_ includes NaNs {((th.square(u - v).tensor))[th.isnan(self.z_)]},{((sq_diff))[th.isnan(self.z_)]},{((sqnorm))[th.isnan(self.z_)]},{((self.x_).tensor)[th.isnan(self.z_)]}, {((self.x_ * (self.x_ + 2.0)).tensor)[th.isnan(self.z_)]}, {((self.x_ * (self.x_ + 2.0)).tensor.sum(-1))[th.isnan(self.z_)]}"
        return th.log1p((self.x_ + self.z_).tensor.sum(-1))

    @staticmethod
    def backward(self, g):
        u, v = self.saved_tensors
        u = MCTensor(u.size()[:-1], nc=u.size(-1), val=u)
        v = MCTensor(v.size()[:-1], nc=v.size(-1), val=v)
        self.z_[self.z_ == 0.0] = th.ones_like(self.z_[self.z_ == 0.0]) * 1e-6
        d = u.size(-1)
        g = g.unsqueeze(-1).expand(u.size())
        gu = th.zeros_like(u)  # m*n*(2d+1)
        gv = th.zeros_like(v)  # m*n*(2d+1)
        auxli_term = th.div(u - v, (u[..., -1] * v[..., -1]).unsqueeze(-2).expand_as(u))  # m*n*d
#         assert th.isnan(auxli_term).max() == 0, f"auxli_term includes NaNs {auxli_term.tensor}"
        gu[..., :-1] = th.div(1, self.z_).unsqueeze(-1) * \
            auxli_term[..., :d-1]  # m*n*(d-1)
#         tmp = th.div(1, self.z_)
#         assert th.isnan(tmp).max() == 0, f"tmp includes NaNs {tmp} {self.z_[th.isnan(tmp)]}"
        gu[..., -1] = th.div(1, self.z_) * (auxli_term[..., -1] -
                                            th.div(self.x_, u[..., -1]))  # m*n
        gv[..., :-1] = -1 * gu[..., :-1]  # m*n*(d-1)
        gv[..., -1] = th.div(1, self.z_) * (-1 * auxli_term[..., - 1] - th.div(self.x_, v[..., -1]))  # m*n
#         assert th.isnan(gu).max() == 0, f"gu includes NaNs {gu.tensor[th.isnan(gu)]}, {auxli_term.tensor[th.isnan(gu)]}"
#         assert th.isnan(gv).max() == 0, f"gv includes NaNs"
        return (g * gu).tensor, (g * gv).tensor
