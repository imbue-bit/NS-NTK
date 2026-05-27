import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class PVR(Optimizer):
    def __init__(self, params, lr=1e-3, sketch_rank=10, oja_lr=1e-2, qr_freq=100):
        defaults = dict(lr=lr, sketch_rank=sketch_rank, oja_lr=oja_lr, qr_freq=qr_freq)
        super().__init__(params, defaults)
        self.state['global'] = {'step': 0, 'U': None}

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            params_with_grad = [p for p in group['params'] if p.grad is not None]
            if not params_with_grad: continue
            
            g = parameters_to_vector([p.grad for p in params_with_grad])
            P, k, g_state = g.numel(), group['sketch_rank'], self.state['global']
            
            if g_state['U'] is None:
                U = torch.randn(P, k, device=g.device, dtype=g.dtype)
                g_state['U'], _ = torch.linalg.qr(U, mode='reduced')
            U = g_state['U']
            
            g_T_U = torch.matmul(g, U)
            V = torch.outer(g, g_T_U)
            U.add_(V - torch.matmul(U, torch.matmul(V.t(), U)), alpha=group['oja_lr'])
            
            g_state['step'] += 1
            if g_state['step'] % group['qr_freq'] == 0:
                g_state['U'], _ = torch.linalg.qr(U, mode='reduced')
                
            g_pvr = g - torch.matmul(U, g_T_U)
            flat_params = parameters_to_vector(params_with_grad)
            flat_params.add_(g_pvr, alpha=-group['lr'])
            vector_to_parameters(flat_params, params_with_grad)
        return loss