"""
Automated Latency & Throughput Benchmark: CPU vs Multi-Core vs Tensor Engine vs OSQP.
"""

import time
import numpy as np
import pandas as pd
import torch

from optiquant.systems.gpu_engine import GPUPortfolioEngine
from optiquant.systems.cpu_baseline import CPUBaselineEngine
from optiquant.optimization.convex_solver import AnalyticalConvexOptimizer


def run_benchmark(num_portfolios: int = 1_000_000, num_assets: int = 15):
    print("=" * 80)
    print(f"OptiQuant-GPU Systems Latency Benchmark ({num_portfolios:,} Portfolios, {num_assets} Assets)")
    print("=" * 80)

    # Synthetic return vector and PSD covariance matrix
    rng = np.random.default_rng(42)
    mu = rng.uniform(0.06, 0.22, num_assets)
    a = rng.normal(0, 1, (num_assets, num_assets))
    cov = np.dot(a, a.T) * 0.05 + np.eye(num_assets) * 0.01

    cpu_engine = CPUBaselineEngine()
    gpu_engine = GPUPortfolioEngine()
    convex = AnalyticalConvexOptimizer(mu, cov)

    # 1. Single-Core Sequential (profile 50,000 and extrapolate to avoid 20+ sec wait)
    seq_sample = 50_000
    seq_res = cpu_engine.run_sequential(mu, cov, num_simulations=seq_sample)
    seq_latency = (seq_res["elapsed_ms"] / seq_sample) * num_portfolios
    seq_throughput = num_portfolios / (seq_latency / 1000.0)

    # 2. Multi-Core Multiprocessing (profile 100,000 and extrapolate)
    mp_sample = 100_000
    mp_res = cpu_engine.run_multiprocessing(mu, cov, num_simulations=mp_sample)
    mp_latency = (mp_res["elapsed_ms"] / mp_sample) * num_portfolios
    mp_throughput = num_portfolios / (mp_latency / 1000.0)

    # 3. Vectorized PyTorch (CPU Tensor Engine)
    cpu_gpu_engine = GPUPortfolioEngine(device="cpu")
    cpu_tensor_res = cpu_gpu_engine.simulate_allocations(mu, cov, num_simulations=num_portfolios)
    cpu_tensor_latency = cpu_tensor_res["elapsed_ms"]
    cpu_tensor_throughput = num_portfolios / (cpu_tensor_latency / 1000.0)

    # 4. OptiQuant-GPU (CUDA or native hardware target)
    gpu_res = gpu_engine.simulate_allocations(mu, cov, num_simulations=num_portfolios)
    gpu_latency = gpu_res["elapsed_ms"]
    gpu_throughput = num_portfolios / (gpu_latency / 1000.0)

    # 5. Exact Analytical Convex Solver
    t0 = time.perf_counter()
    convex.max_sharpe()
    convex.minimum_variance()
    convex.compute_efficient_frontier(points=50)
    convex_latency = (time.perf_counter() - t0) * 1000.0

    print(f"{'Engine / Device':<40} {'Latency (ms)':<15} {'Speedup':<12} {'Throughput (Port/sec)':<20}")
    print("-" * 80)
    dev_str = f"4. OptiQuant-GPU ({gpu_res['device']})"
    print(f"{'1. Python Sequential (Single-Core)':<40} {seq_latency:>9.1f} ms {1.0:>10.1f}x {seq_throughput:>18,.0f}")
    print(f"{'2. Python Multiprocessing (Pool)':<40} {mp_latency:>9.1f} ms {seq_latency / mp_latency:>10.1f}x {mp_throughput:>18,.0f}")
    print(f"{'3. PyTorch Vectorized (CPU)':<40} {cpu_tensor_latency:>9.1f} ms {seq_latency / cpu_tensor_latency:>10.1f}x {cpu_tensor_throughput:>18,.0f}")
    print(f"{dev_str:<40} {gpu_latency:>9.1f} ms {seq_latency / gpu_latency:>10.1f}x {gpu_throughput:>18,.0f}")
    print(f"{'5. Analytical Convex Solver (SLSQP)':<40} {convex_latency:>9.1f} ms {seq_latency / convex_latency:>10.1f}x {'Exact Frontier':>20}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
