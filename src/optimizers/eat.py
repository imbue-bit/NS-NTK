import torch
from torch.optim.optimizer import Optimizer

class EAT(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), epsilon=1e-5):
        defaults = dict(lr=lr, betas=betas, epsilon=epsilon)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2, eps = group['betas'][0], group['betas'][1], group['epsilon']
            global_trace = 1e-8
            
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg_sq'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                global_trace += state['exp_avg_sq'].sum().item()

            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                state['step'] += 1
                state['exp_avg'].mul_(beta1).add_(p.grad, alpha=1 - beta1)
                
                bias1 = 1 - beta1 ** state['step']
                bias2 = 1 - beta2 ** state['step']
                v_norm = (state['exp_avg_sq'] / bias2) / global_trace
                
                update = state['exp_avg'] / (v_norm + eps)
                p.add_(update, alpha=-(group['lr'] / bias1))
        return loss