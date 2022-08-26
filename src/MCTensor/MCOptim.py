from .MCTensor import MCTensor
import torch
import torch.optim
from collections import defaultdict
import math


class MCOptimizer(object):

    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)

        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        return "MCOptimizer"

    def state_dict(self):
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != 'params'}
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in param_mappings})
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def zero_grad(self, set_to_none=False):
        for group in self.param_groups:
            for p in group['params']:
                if isinstance(p, MCTensor):
                    p = p.fc
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

    def step(self):
        raise NotImplementedError

    def add_param_group(self, param_group):
        params = param_group['params']
        if isinstance(params, MCTensor):
            param_group['params'] = [params]
        else:
            param_group['params'] = list(params)

        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError(
                "some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)


class MCSGD(MCOptimizer):

    def check_params(self, params, lr, momentum=0, dampening=0,
                     weight_decay=0, nesterov=False):
        assert lr > 0.0, f"Invalid learning rate: {lr}"
        assert momentum >= 0.0, f"Invalid momentum value: {momentum}"
        assert weight_decay >= 0.0, f"Invalid weight_decay value: {weight_decay}"
        assert not (nesterov and (momentum <= 0 or dampening != 0)), \
            "Nesterov momentum requires a momentum and zero dampening"

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        self.check_params(params, lr, momentum, dampening,
                          weight_decay, nesterov)
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        self.defaults = defaults
        super(MCSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MCSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def sgd(self, params, d_p_list, momentum_buffer_list,
            weight_decay, momentum, lr, dampening, nesterov):

        for i, param in enumerate(params):
            d_p = d_p_list[i]
            if weight_decay != 0:
                d_p = torch.add(d_p, param, alpha=weight_decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            param.add_(d_p, alpha=-lr)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            self.sgd(params_with_grad,
                     d_p_list,
                     momentum_buffer_list,
                     weight_decay=weight_decay,
                     momentum=momentum,
                     lr=lr,
                     dampening=dampening,
                     nesterov=nesterov)

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer


class MCAdam(MCOptimizer):

    def check_params(self, params, lr, betas, eps, weight_decay, amsgrad):
        assert 0.0 < lr, f"Invalid learning rate: {lr}"
        assert 0.0 <= eps, f"Invalid epsilon value: {eps}"
        assert 0.0 <= betas[0] < 1.0, f"Invalid beta parameter at index 0: {betas[0]}"
        assert 0.0 <= betas[1] < 1.0, f"Invalid beta parameter at index 1: {betas[1]}"
        assert 0.0 <= weight_decay, f"Invalid weight_decay value: {weight_decay}"

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        self.check_params(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                          weight_decay=0, amsgrad=False)
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MCAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MCAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def adam(self, params, grads,
             exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
             state_steps, amsgrad,
             beta1, beta2, lr, weight_decay, eps):

        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i],
                              exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() /
                         math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() /
                         math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            self.adam(params_with_grad,
                      grads,
                      exp_avgs,
                      exp_avg_sqs,
                      max_exp_avg_sqs,
                      state_steps,
                      amsgrad=group['amsgrad'],
                      beta1=beta1,
                      beta2=beta2,
                      lr=group['lr'],
                      weight_decay=group['weight_decay'],
                      eps=group['eps'])
