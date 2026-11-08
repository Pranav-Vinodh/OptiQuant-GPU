"""
CPU Execution Baselines (Sequential, Multiprocessing, and NumPy Vectorized).
Used to evaluate systems speedups and IPC overhead against the GPU Tensor Engine.
"""

from typing import Dict, Any, Tuple
import multiprocessing as mp
import time
import numpy as np


def _eval_batch_cpu(args: Tuple[int, np.ndarray, np.ndarray, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Worker task evaluating a batch of random portfolios."""
    batch_size, mu, sigma, rf = args
    n_assets = len(mu)
    raw_w = np.random.uniform(0.0, 1.0, (batch_size, n_assets))
    weights = raw_w / np.sum(raw_w, axis=1, keepdims=True)

    rets = np.dot(weights, mu)
    w_sigma = np.dot(weights, sigma)
    variances = np.sum(w_sigma * weights, axis=1)
    vols = np.sqrt(np.maximum(variances, 1e-8))
    sharpes = (rets - rf) / vols
    return rets, vols, sharpes


class CPUBaselineEngine:
    """
    Standard CPU portfolio evaluation baselines.
    """

    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or max(1, mp.cpu_count())

    def run_sequential(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        num_simulations: int = 100_000,
        risk_free_rate: float = 0.045
    ) -> Dict[str, Any]:
        """Single-threaded sequential evaluation."""
        mu = np.asarray(mean_returns, dtype=np.float64)
        sigma = np.asarray(cov_matrix, dtype=np.float64)
        n_assets = len(mu)

        start = time.perf_counter()
        rets = np.zeros(num_simulations)
        vols = np.zeros(num_simulations)
        sharpes = np.zeros(num_simulations)

        for i in range(num_simulations):
            w = np.random.random(n_assets)
            w /= np.sum(w)
            r = float(np.sum(mu * w))
            v = float(np.sqrt(max(np.dot(w.T, np.dot(sigma, w)), 1e-8)))
            s = (r - risk_free_rate) / v
            rets[i] = r
            vols[i] = v
            sharpes[i] = s

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "elapsed_ms": elapsed_ms,
            "engine": "Sequential CPU",
            "num_simulations": num_simulations,
            "throughput": num_simulations / max(elapsed_ms / 1000.0, 1e-6)
        }

    def run_multiprocessing(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        num_simulations: int = 100_000,
        risk_free_rate: float = 0.045
    ) -> Dict[str, Any]:
        """Multi-core CPU pool with IPC overhead."""
        mu = np.asarray(mean_returns, dtype=np.float64)
        sigma = np.asarray(cov_matrix, dtype=np.float64)

        chunk_size = num_simulations // self.num_workers
        remainder = num_simulations % self.num_workers

        batches = []
        for i in range(self.num_workers):
            count = chunk_size + (remainder if i == self.num_workers - 1 else 0)
            if count > 0:
                batches.append((count, mu, sigma, risk_free_rate))

        start = time.perf_counter()
        with mp.Pool(processes=self.num_workers) as pool:
            pool.map(_eval_batch_cpu, batches)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "elapsed_ms": elapsed_ms,
            "engine": f"Multiprocessing CPU ({self.num_workers} cores)",
            "num_simulations": num_simulations,
            "throughput": num_simulations / max(elapsed_ms / 1000.0, 1e-6)
        }
