"""
Vectorized PyTorch Tensor Kernel for High-Throughput Monte Carlo Portfolio Simulation.
"""

from typing import Dict, Any, Union
import time
import numpy as np
import pandas as pd
import torch


class GPUPortfolioEngine:
    """
    Executes batched Monte Carlo portfolio evaluation entirely in GPU VRAM (or CPU tensor vectorization).
    Eliminates Python IPC pickling overhead and GIL bottlenecks.
    """

    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def simulate_allocations(
        self,
        mean_returns: Union[torch.Tensor, np.ndarray, pd.Series],
        cov_matrix: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        num_simulations: int = 1_000_000,
        risk_free_rate: float = 0.045
    ) -> Dict[str, Any]:
        """
        Simulates 1,000,000+ random portfolio allocations in parallel on tensor cores.
        """
        # Convert inputs to torch float32 on target device
        if isinstance(mean_returns, (pd.Series, pd.DataFrame)):
            mean_returns = mean_returns.values
        if isinstance(cov_matrix, pd.DataFrame):
            cov_matrix = cov_matrix.values

        if not isinstance(mean_returns, torch.Tensor):
            mu = torch.from_numpy(np.asarray(mean_returns, dtype=np.float32)).to(self.device)
        else:
            mu = mean_returns.to(device=self.device, dtype=torch.float32)

        if not isinstance(cov_matrix, torch.Tensor):
            sigma = torch.from_numpy(np.asarray(cov_matrix, dtype=np.float32)).to(self.device)
        else:
            sigma = cov_matrix.to(device=self.device, dtype=torch.float32)

        num_assets = mu.shape[0]

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # 1. Generate uniform random simplex weights: sum(w) = 1
        raw_weights = torch.rand((num_simulations, num_assets), device=self.device, dtype=torch.float32)
        weights = raw_weights / torch.sum(raw_weights, dim=1, keepdim=True)

        # 2. Vectorized expected returns: R_p = W * mu
        port_returns = torch.matmul(weights, mu)

        # 3. Vectorized portfolio variance: sigma_p^2 = sum((W * Sigma) * W, dim=1)
        w_sigma = torch.matmul(weights, sigma)
        port_variances = torch.sum(w_sigma * weights, dim=1)
        port_volatilities = torch.sqrt(torch.clamp(port_variances, min=1e-8))

        # 4. Vectorized Sharpe ratios
        sharpe_ratios = (port_returns - risk_free_rate) / port_volatilities

        # 5. Extract optimal points via GPU reduction kernels
        max_sharpe_idx = torch.argmax(sharpe_ratios)
        min_vol_idx = torch.argmin(port_volatilities)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Downsample for responsive UI rendering
        sample_size = min(10_000, num_simulations)
        scatter_returns = port_returns[:sample_size].detach().cpu().numpy()
        scatter_vols = port_volatilities[:sample_size].detach().cpu().numpy()
        scatter_sharpes = sharpe_ratios[:sample_size].detach().cpu().numpy()

        return {
            "elapsed_ms": elapsed_ms,
            "device": str(self.device),
            "num_simulations": num_simulations,
            "max_sharpe": {
                "return": float(port_returns[max_sharpe_idx].item()),
                "volatility": float(port_volatilities[max_sharpe_idx].item()),
                "sharpe": float(sharpe_ratios[max_sharpe_idx].item()),
                "weights": weights[max_sharpe_idx].detach().cpu().numpy()
            },
            "min_volatility": {
                "return": float(port_returns[min_vol_idx].item()),
                "volatility": float(port_volatilities[min_vol_idx].item()),
                "sharpe": float(sharpe_ratios[min_vol_idx].item()),
                "weights": weights[min_vol_idx].detach().cpu().numpy()
            },
            "scatter_sample": {
                "returns": scatter_returns,
                "volatilities": scatter_vols,
                "sharpe": scatter_sharpes
            }
        }
