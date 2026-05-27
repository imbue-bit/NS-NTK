import torch
import torch.nn as nn
from torch.func import functional_call, vmap, jacrev
from typing import Tuple

class NTKSpectralAnalyzer:
    """
    Theoretical tool to compute the Empirical NTK and project target signals onto its eigenbasis.
    Validates Proposition 1 (Spectral Concentration of Smooth Factors).
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.params = dict(model.named_parameters())

    def _forward_single(self, params, x):
        return functional_call(self.model, params, (x.unsqueeze(0),)).squeeze()

    @torch.no_grad()
    def compute_spectrum_and_projection(self, X: torch.Tensor, target_signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the NTK eigenvalues and the energy projection of a specific signal.
        
        Args:
            X: Input features (N, D).
            target_signal: The target vector (e.g., f_\beta or alpha) of shape (N,).
            
        Returns:
            eigenvalues: NTK eigenvalues sorted descending (N,).
            signal_energy: Normalized projected energy on each eigenvector (N,).
        """
        # 1. Compute Jacobian J via vmap and jacrev
        jac_dict = vmap(jacrev(self._forward_single), (None, 0))(self.params, X)
        
        # Flatten and concatenate Jacobians
        J_flat = [j.reshape(X.shape[0], -1) for j in jac_dict.values()]
        J = torch.cat(J_flat, dim=1)  # Shape: (N, P)
        
        # 2. Construct empirical NTK matrix: Theta = J @ J^T
        NTK = torch.matmul(J, J.t())  # Shape: (N, N)
        
        # 3. Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(NTK)
        
        # Sort descending
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 4. Sobolev-space projection (Dirichlet energy calculation equivalent)
        # <f, \phi_i>^2
        energy = torch.matmul(target_signal.unsqueeze(0), eigenvectors).squeeze() ** 2
        energy_normalized = energy / (energy.sum() + 1e-8)
        
        return eigenvalues, energy_normalized