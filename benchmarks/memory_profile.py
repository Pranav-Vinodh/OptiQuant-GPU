"""
CUDA Memory Profiler and Tensor Bandwidth Analysis.
"""

import torch
import numpy as np
from optiquant.systems.gpu_engine import GPUPortfolioEngine


def profile_memory():
    print("=" * 70)
    print("OptiQuant-GPU Memory & VRAM Allocation Profile")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")
    if device.type == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
        torch.cuda.reset_peak_memory_stats()

    n_assets = 20
    n_sims = 1_000_000

    mu = np.random.uniform(0.05, 0.20, n_assets)
    cov = np.eye(n_assets) * 0.04

    engine = GPUPortfolioEngine(device=str(device))
    res = engine.simulate_allocations(mu, cov, num_simulations=n_sims)

    # Tensor size in memory: (1,000,000 x 20) float32 = 20M floats = 80 MB
    raw_weights_mb = (n_sims * n_assets * 4) / (1024**2)
    returns_mb = (n_sims * 4) / (1024**2)
    vols_mb = (n_sims * 4) / (1024**2)
    sharpes_mb = (n_sims * 4) / (1024**2)
    total_data_mb = raw_weights_mb + returns_mb + vols_mb + sharpes_mb

    print(f"Simulated Portfolios: {n_sims:,}")
    print(f"Number of Assets: {n_assets}")
    print(f"Execution Latency: {res['elapsed_ms']:.2f} ms")
    print(f"Theoretical Active Tensor Footprint: {total_data_mb:.2f} MB")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
        print(f"Peak VRAM Allocated: {peak_mb:.2f} MB")
        print(f"Peak VRAM Reserved: {reserved_mb:.2f} MB")
        effective_bandwidth_gbps = (total_data_mb / (1024)) / (res['elapsed_ms'] / 1000.0)
        print(f"Effective Memory Bandwidth: {effective_bandwidth_gbps:.2f} GB/s")
    else:
        print("Running in CPU tensor vector mode (zero VRAM overhead).")

    print("=" * 70)


if __name__ == "__main__":
    profile_memory()
