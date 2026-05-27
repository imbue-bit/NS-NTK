import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

class ExponentialDriftSampler:
    """
    Temporal Generalization Bound with Exponential Aggregation.
    """
    def __init__(self, timestamps: np.ndarray):
        """
        Args:
            timestamps: 1D array of temporal indices or real timestamps for the dataset.
        """
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        self.t_now = self.timestamps.max()
        self.time_lags = self.t_now - self.timestamps

    def compute_optimal_lambda(self, arrival_rate_c: float, drift_gamma: float, 
                               L_ell: float = 1.0, L_f: float = 1.0, C_F: float = 1.0) -> float:
        """
        Computes analytical optimal decay rate \lambda^* (Equation 4).
        """
        numerator = arrival_rate_c * (L_ell * L_f * drift_gamma) ** 2
        return (numerator / C_F) ** (1/3)

    def get_sampler(self, arrival_rate_c: float, drift_gamma: float, **kwargs) -> WeightedRandomSampler:
        """
        Returns a PyTorch WeightedRandomSampler that samples from the aggregated measure \bar{\mathcal{D}}_\lambda.
        """
        lambda_star = self.compute_optimal_lambda(arrival_rate_c, drift_gamma, **kwargs)
        
        # Exponential weights: w \propto \lambda * exp(-\lambda * s)
        weights = lambda_star * np.exp(-lambda_star * self.time_lags)
        weights_normalized = weights / weights.sum()
        
        return WeightedRandomSampler(
            weights=torch.tensor(weights_normalized, dtype=torch.float64), 
            num_samples=len(self.timestamps), 
            replacement=True
        )