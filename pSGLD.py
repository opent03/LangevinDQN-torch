import torch
from torch.optim.optimizer import Optimizer
import math

class pSGLD_Adam(Optimizer):
    ''' Implements Langevin SGD updates to params '''

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(pSGLD_Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(pSGLD_Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['mt'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['vt'] = torch.ones_like(p, memory_format=torch.preserve_format)

                mt, vt = state['mt'], state['vt']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                mt.mul_(beta1).add_(1-beta1, grad)
                vt.mul_(beta2).addcmul_(1-beta2, grad, grad)

                preconditioner = (vt.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                sigma = 1. / torch.sqrt(torch.tensor(group['lr']))            # correction term
                langevin_noise = torch.normal(
                                            mean= torch.zeros_like(grad),
                                            std=torch.ones_like(grad)
                                            ) * sigma * torch.sqrt(preconditioner)
                delta_p = 0.5 * preconditioner * mt * (1./bias_correction1) + langevin_noise

                p.data.add_(delta_p, alpha=-group['lr'])
        return loss

class pSGLD_RMSprop(Optimizer):
    ''' Borrowed from pysgmcmc's documentation for sanity checking '''
    def __init__(self,
                 params,
                 lr=1e-2,
                 precondition_decay_rate=0.95,
                 num_pseudo_batches=1,
                 num_burn_in_steps=3000,
                 diagonal_bias=1e-8):

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr, precondition_decay_rate=precondition_decay_rate,
            num_pseudo_batches=num_pseudo_batches,
            num_burn_in_steps=num_burn_in_steps,
            diagonal_bias=1e-8,
        )
        super().__init__(params, defaults)

    def step(self):
        loss = None
        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr = group["lr"]
                num_pseudo_batches = group["num_pseudo_batches"]
                precondition_decay_rate = group["precondition_decay_rate"]
                gradient = parameter.grad.data

                #  State initialization {{{ #

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.ones_like(parameter)

                #  }}} State initialization #

                state["iteration"] += 1

                momentum = state["momentum"]

                #  Momentum update {{{ #
                momentum.add_(
                    (1.0 - precondition_decay_rate) * ((gradient ** 2) - momentum)
                )
                #  }}} Momentum update #

                if state["iteration"] > group["num_burn_in_steps"]:
                    sigma = 1. / torch.sqrt(torch.tensor(lr))
                else:
                    sigma = torch.zeros_like(parameter)

                preconditioner = (
                        1. / torch.sqrt(momentum + group["diagonal_bias"])
                )

                scaled_grad = (
                        0.5 * preconditioner * gradient * num_pseudo_batches +
                        torch.normal(
                            mean=torch.zeros_like(gradient),
                            std=torch.ones_like(gradient)
                        ) * sigma * torch.sqrt(preconditioner)
                )

                parameter.data.add_(-lr * scaled_grad)

        return loss