r"""transfer_matrix.py

Cluster-Level Capability Transfer Matrix Engine for VGCL Phase 0.

Estimates cluster-level capability transfer across intervention failure clusters C_k
and evaluation clusters E_j:
    M_{k, j} = P(Solve(E_j) | \theta + \Delta \theta(C_k)) - P(Solve(E_j) | \theta)

Computes within-cluster, cross-cluster, average transfer, and 95% confidence intervals.
"""

import json
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


CLUSTERS = [
    "FC_01_MICRO_CORRECTNESS",
    "FC_02_SCALE_DIVERGENCE",
    "FC_03_TERMINATION",
    "FC_04_BOUNDS_VIOLATION",
    "FC_05_RESOURCE_FAILURE",
    "FC_06_FORMAT_CONTRACT",
    "FC_07_STATE_INCONSISTENCY",
    "FC_08_UNKNOWN",
]


@dataclass
class TransferResults:
    matrix: np.ndarray  # Shape (K, J)
    within_cluster: Dict[str, float]
    cross_cluster_avg: float
    overall_avg: float
    ci_lower: np.ndarray
    ci_upper: np.ndarray


def compute_confidence_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson score 95% confidence interval for pass rate binomial distributions."""
    if trials == 0:
        return 0.0, 0.0
    p_hat = successes / trials
    z = 1.96  # 95% confidence
    denom = 1 + (z**2) / trials
    center = (p_hat + (z**2) / (2 * trials)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * trials)) / trials) / denom
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return lower, upper


class TransferMatrixCalculator:
    """Computes cluster-level transfer matrices and statistical summaries."""

    def __init__(self, cluster_list: Optional[List[str]] = None):
        self.clusters = cluster_list or CLUSTERS
        self.num_clusters = len(self.clusters)

    def calculate_transfer_matrix(
        self,
        base_pass_rates: Dict[str, float],
        ft_pass_rates: Dict[str, Dict[str, float]],
        eval_trials: int = 40,
    ) -> TransferResults:
        """
        base_pass_rates: Dict[eval_cluster, float] -> P(Solve(E_j) | theta)
        ft_pass_rates: Dict[intervention_cluster, Dict[eval_cluster, float]] -> P(Solve(E_j) | theta + delta theta(C_k))
        """
        K = self.num_clusters
        J = self.num_clusters
        matrix = np.zeros((K, J))
        ci_lower = np.zeros((K, J))
        ci_upper = np.zeros((K, J))

        within_cluster: Dict[str, float] = {}

        for k_idx, c_k in enumerate(self.clusters):
            ft_cluster_evals = ft_pass_rates.get(c_k, {})
            for j_idx, e_j in enumerate(self.clusters):
                p_base = base_pass_rates.get(e_j, 0.0)
                p_ft = ft_cluster_evals.get(e_j, p_base)
                delta = p_ft - p_base
                matrix[k_idx, j_idx] = delta

                # Calculate confidence interval for delta
                succ_ft = int(round(p_ft * eval_trials))
                low, high = compute_confidence_interval(succ_ft, eval_trials)
                ci_lower[k_idx, j_idx] = low - p_base
                ci_upper[k_idx, j_idx] = high - p_base

                if k_idx == j_idx:
                    within_cluster[c_k] = delta

        # Cross-cluster transfer (off-diagonal elements)
        mask = ~np.eye(K, dtype=bool)
        cross_cluster_avg = float(np.mean(matrix[mask])) if K > 1 else 0.0
        overall_avg = float(np.mean(matrix))

        return TransferResults(
            matrix=matrix,
            within_cluster=within_cluster,
            cross_cluster_avg=cross_cluster_avg,
            overall_avg=overall_avg,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    def print_transfer_summary(self, results: TransferResults) -> None:
        print("\n========================================================")
        print("    VGCL CLUSTER-LEVEL CAPABILITY TRANSFER MATRIX (M_{k,j})")
        print("========================================================")
        header = f"{'Intervention (C_k)':<26} | " + " | ".join([c[:5] for c in self.clusters])
        print(header)
        print("-" * len(header))

        for k_idx, c_k in enumerate(self.clusters):
            row_str = f"{c_k:<26} | "
            row_vals = [f"{results.matrix[k_idx, j_idx]:+6.2f}" for j_idx in range(self.num_clusters)]
            row_str += " | ".join(row_vals)
            print(row_str)

        print("\n--------------------------------------------------------")
        print(f"  Within-Cluster Transfer Avg:  {np.mean(list(results.within_cluster.values())):+6.4f}")
        print(f"  Cross-Cluster Transfer Avg:   {results.cross_cluster_avg:+6.4f}")
        print(f"  Overall Average Transfer:     {results.overall_avg:+6.4f}")
        print("========================================================\n")


if __name__ == "__main__":
    calc = TransferMatrixCalculator()
    # Baseline pass rates on base model theta
    base_rates = {c: 0.15 for c in CLUSTERS}
    # Simulated fine-tuned pass rates
    ft_rates = {
        c_k: {e_j: (0.45 if c_k == e_j else 0.20) for e_j in CLUSTERS}
        for c_k in CLUSTERS
    }

    res = calc.calculate_transfer_matrix(base_rates, ft_rates)
    calc.print_transfer_summary(res)
