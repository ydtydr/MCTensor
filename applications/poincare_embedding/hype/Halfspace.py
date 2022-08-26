#!/usr/bin/env python3
import torch as th
from torch.autograd import Function
from .common import acosh
from .manifold import Manifold


class HalfspaceManifold(Manifold):
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
        d = u.size(-1)
        uu = th.zeros(u.size(0), d + 1)
        if u.is_cuda:
            uu = uu.cuda()
        squnom = th.sum(th.pow(u, 2), dim=-1)  # n
        uu[..., 0] = th.div(th.ones_like(u[..., -1]), u[..., -1]
                            ) + th.div(squnom, 4 * u[..., -1])  # n
        uu[..., 1] = th.div(th.ones_like(u[..., -1]), u[..., -1]
                            ) - th.div(squnom, 4 * u[..., -1])  # n
        uu[..., 2:] = th.div(
            u[..., :d - 1], u[..., -1].unsqueeze(-1).expand_as(u[..., :d - 1]))
        return uu.narrow(-1, 1, d) / (uu.narrow(-1, 0, 1) + 1)

    def distance(self, uu, vv):
        dis = HalfspaceDistance.apply(uu, vv)
        #dis = HalfspaceDistance(uu, vv)
        return dis

    def pnorm(self, u):
        return th.sqrt(th.sum(th.pow(self.to_poincare_ball(u), 2), dim=-1))

    def normalize(self, w):
        """Normalize vector such that it is located on the halfspace model"""
#         d = w.size(-1)
        narrowed = w.narrow(-1, -1, 1)
        narrowed.clamp_(min=0.0)
        return w

    def init_weights(self, w, irange=1e-5):
        th.manual_seed(40)
        w.data.zero_()
        #########
        d = w.size(-1)
        w.data[..., :-1].uniform_(-irange, irange)
        w.data[..., -1] = 1.0 + irange * 2 * (th.rand_like(w[..., -1])-0.5)

    def rgrad(self, p, d_p):
        """Euclidean gradient for halfspace"""
        if d_p.is_sparse:
            u = d_p._values()
            x = p.index_select(0, d_p._indices().squeeze())
        else:
            u = d_p
            x = p
        # transform from Euclidean grad to Riemannian grad
        u.mul_((x[..., -1]).unsqueeze(-1))
        return d_p

    def expm(self, p, d_p, lr=None, out=None, normalize=False):
        """Exponential map for halfspace model"""
        d = p.size(-1)
        if out is None:
            out = p
        if d_p.is_sparse:
            ix, d_val = d_p._indices().squeeze(), d_p._values()
            p_val = self.normalize(p.index_select(0, ix))
            newp_val = p_val.clone()
            #######===================================================########
            # Original Exponential map arithmetic
            s = th.norm(d_val,dim=-1)#n
            zeros_mask = (s == 0.0)
            coshs = th.cosh(s)
            sihncs = self.sinhc(s)
            sihncs[zeros_mask] = th.ones_like(sihncs[th.isnan(sihncs)])
#             sihncs[th.isnan(sihncs)] = th.ones_like(sihncs[th.isnan(sihncs)])
            assert th.isnan(coshs).max()==0, "coshs includes NaNs"
            assert th.isnan(sihncs).max()==0, "sihncs includes NaNs"
            newp_val[...,:-1] = p_val[...,:-1] + th.div(p_val[...,-1], th.div(coshs, sihncs)-d_val[...,-1]).unsqueeze(-1) * d_val[...,:-1]#n*(d-1)
#             newp_val[...,-1] = p_val[...,-1] * th.div(s, s*coshs-d_val[...,-1]*th.sinh(s))#n
            newp_val[...,-1] = th.div(p_val[...,-1], coshs-d_val[...,-1]*sihncs) 
            newp_val = self.normalize(newp_val)
            p.index_copy_(0, ix, newp_val)
        else:
            raise NotImplementedError


class HalfspaceDistance(Function):
    @staticmethod
    def forward(self, u, v, myeps=0.0):
        assert th.isnan(u).max() == 0, "u includes NaNs"
        assert th.isnan(v).max() == 0, "v includes NaNs"
        if len(u) < len(v):
            u = u.expand_as(v)
        elif len(u) > len(v):
            v = v.expand_as(u)
        d = u.size(-1)
        self.save_for_backward(u, v)
        sqnorm = th.sum(th.square(u - v), dim=-1)
        self.x_ = th.div(sqnorm, 2.0*u[..., -1] * v[..., -1])  # m*n
        self.z_ = th.sqrt(self.x_ * (self.x_ + 2.0))  # m*n
        return th.log1p(self.x_ + self.z_)

    @staticmethod
    def backward(self, g):
        u, v = self.saved_tensors
        self.z_[self.z_ == 0.0] = th.ones_like(self.z_[self.z_ == 0.0]) * 1e-6
        d = u.size(-1)
        g = g.unsqueeze(-1).expand_as(u)
        gu = th.zeros_like(u)  # m*n*(2d+1)
        gv = th.zeros_like(v)  # m*n*(2d+1)
        auxli_term = th.div(
            u - v, (u[..., -1] * v[..., -1]).unsqueeze(-1).expand_as(u))  # m*n*d
        gu[..., :-1] = th.div(1, self.z_).unsqueeze(-1) * \
            auxli_term[..., :d-1]  # m*n*(d-1)
        gu[..., -1] = th.div(1, self.z_) * (auxli_term[..., -1] -
                                            th.div(self.x_, u[..., -1]))  # m*n
        gv[..., :-1] = -1 * gu[..., :-1]  # m*n*(d-1)
        gv[..., -1] = th.div(1, self.z_) * (-1 * auxli_term[..., -
                                                            1] - th.div(self.x_, v[..., -1]))  # m*n
        assert th.isnan(gu).max() == 0, "gu includes NaNs"
        assert th.isnan(gv).max() == 0, "gv includes NaNs"
        return g * gu, g * gv
