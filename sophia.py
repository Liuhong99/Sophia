import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional


class SophiaG(Optimizer):
    def __init__(self, params, lr=1e-5, betas=(0.965, 0.99), gamma = 0.04,
         weight_decay=1e-1, *, maximize: bool = False,
         capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= gamma:
            raise ValueError("Invalid gamma parameter at index 1: {}".format(gamma))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, gamma=gamma, 
                        weight_decay=weight_decay, 
                        maximize=maximize, capturable=capturable)
        super(SophiaG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))
    
    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                
                if p.grad.is_sparse:
                    raise RuntimeError('Hero does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)                

                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])
                hessian.append(state['hessian'])

            sophiag(params_with_grad,
                  grads,
                  exp_avgs,
                  hessian,
                  state_steps,
                  beta1=beta1,
                  beta2=beta2,
                  gamma=group['gamma'],
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  maximize=group['maximize'],
                  capturable=group['capturable'])

        return loss

def sophiag(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          hessian: List[Tensor],
          state_steps: List[Tensor],
          capturable: bool = False,
          *,
          beta1: float,
          beta2: float,
          gamma: float,
          lr: float,
          weight_decay: float,
          maximize: bool):

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    
    func = _single_tensor_sophiag

    func(params,
         grads,
         exp_avgs,
         hessian,
         state_steps,
         beta1=beta1,
         beta2=beta2,
         gamma=gamma,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize,
         capturable=capturable)

def _single_tensor_sophiag(params: List[Tensor],
                         grads: List[Tensor],
                         exp_avgs: List[Tensor],
                         hessian: List[Tensor],
                         state_steps: List[Tensor],
                         *,
                         beta1: float,
                         beta2: float,
                         gamma: float,
                         lr: float,
                         weight_decay: float,
                         maximize: bool,
                         capturable: bool):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        if capturable:
            step = step_t
            step_size = lr 
            step_size_neg = step_size.neg()

            ratio = (exp_avg.abs() / (gamma * hess + 1e-15)).clamp(None,1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
        else:
            step = step_t.item()
            step_size_neg = - lr 
            
            ratio = (exp_avg.abs() / (gamma * hess + 1e-15)).clamp(None,1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)

class SophiaH(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), gamma = 10,
         weight_decay=1e-2, *, maximize: bool = False,
         capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= gamma:
            raise ValueError("Invalid gamma parameter at index 1: {}".format(gamma))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, gamma=gamma, 
                        weight_decay=weight_decay, 
                        maximize=maximize, capturable=capturable)
        super(SophiaH, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))
    
    @torch.no_grad()
    def update_hessian(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['hessian'].mul_(beta2).add_(p.grad, alpha=1 - beta2)
                
        return loss
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                
                if p.grad.is_sparse:
                    raise RuntimeError('Hero does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)                

                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])
                hessian.append(state['hessian'])

            sophiah(params_with_grad,
                  grads,
                  exp_avgs,
                  hessian,
                  state_steps,
                  beta1=beta1,
                  beta2=beta2,
                  gamma=group['gamma'],
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  maximize=group['maximize'],
                  capturable=group['capturable'])

        return loss

def sophiah(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          hessian: List[Tensor],
          state_steps: List[Tensor],
          # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
          # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
          capturable: bool = False,
          *,
          beta1: float,
          beta2: float,
          gamma: float,
          lr: float,
          weight_decay: float,
          maximize: bool):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    
    func = _single_tensor_sophiah

    func(params,
         grads,
         exp_avgs,
         hessian,
         state_steps,
         beta1=beta1,
         beta2=beta2,
         gamma=gamma,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize,
         capturable=capturable)

def _single_tensor_sophiah(params: List[Tensor],
                         grads: List[Tensor],
                         exp_avgs: List[Tensor],
                         hessian: List[Tensor],
                         state_steps: List[Tensor],
                         *,
                         beta1: float,
                         beta2: float,
                         gamma: float,
                         lr: float,
                         weight_decay: float,
                         maximize: bool,
                         capturable: bool):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        step_t += 1

        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        

        if capturable:
            step = step_t
            bias_correction = 1 - torch.pow(beta1, step)

            step_size = lr / bias_correction
            step_size_neg = step_size.neg()

            ratio = (exp_avg.abs() / (gamma * torch.nn.functional.relu(hess) + 1e-12)).clamp(None,1)
            param.add_(exp_avg.sign() * ratio, alpha=step_size_neg)
        else:
            step = step_t.item()

            bias_correction = 1 - beta1 ** (step) 
            step_size_neg = - lr / bias_correction
            
            ratio = (exp_avg.abs() / (gamma * torch.nn.functional.relu(hess) + 1e-12)).clamp(None,1)
            param.add_(exp_avg.sign() * ratio, alpha=step_size_neg)