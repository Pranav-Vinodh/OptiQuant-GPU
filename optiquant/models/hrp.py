"""
Hierarchical Risk Parity (HRP) Portfolio Allocation Engine.
Reference: Marcos López de Prado, "Building Diversified Portfolios that Outperform Out-of-Sample", 2016.
"""

from typing import List
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform


class HierarchicalRiskParity:
    """
    Implements Hierarchical Risk Parity to eliminate matrix inversion singularity (kappa(Sigma) >> 10^3).
    """

    def __init__(self, cov_matrix: pd.DataFrame, corr_matrix: pd.DataFrame):
        self.cov = cov_matrix
        self.corr = corr_matrix
        self.assets = list(cov_matrix.columns)

    def _correl_distance(self) -> np.ndarray:
        """Computes condensed angular correlation distance matrix."""
        corr_vals = np.clip(self.corr.values, -1.0, 1.0)
        dist = np.sqrt(np.clip((1.0 - corr_vals) / 2.0, 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)
        return squareform(dist)

    @staticmethod
    def _get_quasi_diag(link: np.ndarray) -> List[int]:
        """Performs quasi-diagonalization by extracting leaf order from linkage tree."""
        root = to_tree(link, rd=False)
        order = []

        def dfs(node):
            if node.is_leaf():
                order.append(node.id)
            else:
                dfs(node.left)
                dfs(node.right)

        dfs(root)
        return order

    def _get_cluster_var(self, cluster_indices: List[int]) -> float:
        """Calculates cluster variance using inverse-variance intra-cluster weights."""
        sub_cov = self.cov.iloc[cluster_indices, cluster_indices].values
        inv_diag = 1.0 / np.clip(np.diag(sub_cov), 1e-8, None)
        weights = inv_diag / np.sum(inv_diag)
        c_var = float(np.dot(weights.T, np.dot(sub_cov, weights)))
        return max(c_var, 1e-8)

    def solve(self) -> pd.Series:
        """
        Executes hierarchical clustering, quasi-diagonalization, and recursive bisection.
        Returns asset weights summing to 1.0.
        """
        dist_condensed = self._correl_distance()
        link = linkage(dist_condensed, method="single")
        sort_order = self._get_quasi_diag(link)
        ordered_assets = [self.assets[i] for i in sort_order]

        weights = pd.Series(1.0, index=ordered_assets)
        cluster_list = [ordered_assets]

        while len(cluster_list) > 0:
            cluster_list = [
                cluster[start:end]
                for cluster in cluster_list
                for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                if len(cluster) > 1
            ]
            for i in range(0, len(cluster_list), 2):
                c1 = cluster_list[i]
                c2 = cluster_list[i + 1]
                idx1 = [self.assets.index(a) for a in c1]
                idx2 = [self.assets.index(a) for a in c2]
                v1 = self._get_cluster_var(idx1)
                v2 = self._get_cluster_var(idx2)
                alpha = 1.0 - (v1 / (v1 + v2))
                weights[c1] *= alpha
                weights[c2] *= (1.0 - alpha)

        # Normalize and reorder back to original asset column ordering
        weights = weights / weights.sum()
        return weights[self.assets]
