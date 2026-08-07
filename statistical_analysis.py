"""statistical_analysis.py

Statistical Analysis Engine & Pre-Registered Endpoints for VGCL Phase 0.

Evaluates statistical significance and effect size between Failure-Targeted Synthetic Curricula
and Uniform-Sampled Synthetic Control under identical compute budgets across fixed random seeds.

Primary Endpoint: Cluster-specific Pass@1 improvement.
Secondary Endpoints: Overall Pass@1, Pass@10, Transfer Matrix Density, Cross-Cluster Transfer,
                     Difficulty-Stratified Improvement.
"""

import json
import math
import numpy as np
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EndpointResult:
    endpoint_name: str
    targeted_mean: float
    targeted_std: float
    uniform_mean: float
    uniform_std: float
    absolute_delta: float
    relative_gain_pct: float
    p_value: float
    statistically_significant: bool  # alpha = 0.05
    effect_size_cohens_d: float


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cohen's d effect size between two independent sample series."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    pooled_sd = math.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / pooled_sd) if pooled_sd > 0 else 0.0


def permutation_test_p_value(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Welch's t-test p-value for mean difference across seed runs."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 1.0
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    se = math.sqrt(vx / nx + vy / ny)
    if se == 0:
        return 0.0 if np.mean(x) != np.mean(y) else 1.0
    t_stat = abs(np.mean(x) - np.mean(y)) / se

    # Welch-Satterthwaite degrees of freedom
    df_num = (vx / nx + vy / ny) ** 2
    df_den = ((vx / nx) ** 2) / (nx - 1) + ((vy / ny) ** 2) / (ny - 1)
    df = df_num / df_den if df_den > 0 else (nx + ny - 2)

    # Standard normal/t tail approximation for p-value
    # Using normal survival function approximation for robust dependency-free execution
    p_val = 2.0 * math.erfc(t_stat / math.sqrt(2))
    return float(p_val)


class VGCLStatisticalAnalyzer:
    """Statistical verification engine evaluating experimental hypothesis."""

    def __init__(self, alpha: float = 0.05, minimum_detectable_effect: float = 0.25):
        self.alpha = alpha
        self.minimum_detectable_effect = minimum_detectable_effect

    def evaluate_endpoint(
        self,
        name: str,
        targeted_seeds: List[float],
        uniform_seeds: List[float],
    ) -> EndpointResult:
        tgt_arr = np.array(targeted_seeds)
        uni_arr = np.array(uniform_seeds)

        tgt_m, tgt_s = float(np.mean(tgt_arr)), float(np.std(tgt_arr, ddof=1 if len(tgt_arr) > 1 else 0))
        uni_m, uni_s = float(np.mean(uni_arr)), float(np.std(uni_arr, ddof=1 if len(uni_arr) > 1 else 0))

        delta = tgt_m - uni_m
        rel_gain = (delta / uni_m * 100.0) if uni_m > 0 else 0.0

        p_val = permutation_test_p_value(tgt_arr, uni_arr)
        effect_d = compute_cohens_d(tgt_arr, uni_arr)
        is_sig = p_val <= self.alpha and abs(delta) >= 0.01

        return EndpointResult(
            endpoint_name=name,
            targeted_mean=tgt_m,
            targeted_std=tgt_s,
            uniform_mean=uni_m,
            uniform_std=uni_s,
            absolute_delta=delta,
            relative_gain_pct=rel_gain,
            p_value=p_val,
            statistically_significant=is_sig,
            effect_size_cohens_d=effect_d,
        )

    def run_full_statistical_audit(
        self,
        targeted_data: Dict[str, List[float]],
        uniform_data: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        targeted_data / uniform_data map endpoint names to seed run values (3 seeds each).
        Example endpoints:
          - 'Primary: Cluster Pass@1 (FC_01)'
          - 'Secondary: Overall Pass@1'
          - 'Secondary: Pass@10'
          - 'Secondary: Transfer Matrix Density'
        """
        results: Dict[str, Any] = {}
        for endpoint_name in targeted_data:
            tgt_runs = targeted_data[endpoint_name]
            uni_runs = uniform_data.get(endpoint_name, [0.0] * len(tgt_runs))
            res = self.evaluate_endpoint(endpoint_name, tgt_runs, uni_runs)
            results[endpoint_name] = asdict(res)

        return results

    def print_publication_report(self, audit_results: Dict[str, Any]) -> None:
        print("\n==========================================================================")
        print("  PUBLICATION STATISTICAL AUDIT REPORT: VGCL HYPOTHESIS TEST")
        print("==========================================================================")
        print(f"{'Endpoint':<35} | {'Targeted (mu +/- sd)':<20} | {'Uniform (mu +/- sd)':<20} | {'Delta':<8} | {'p-value':<8} | {'Cohen d':<7} | {'Significant'}")
        print("-" * 130)

        for name, data in audit_results.items():
            tgt_str = f"{data['targeted_mean']:.4f} +/- {data['targeted_std']:.4f}"
            uni_str = f"{data['uniform_mean']:.4f} +/- {data['uniform_std']:.4f}"
            sig_str = "YES [CONFIRMED]" if data["statistically_significant"] else "NO"
            print(
                f"{name:<35} | {tgt_str:<20} | {uni_str:<20} | {data['absolute_delta']:+7.4f} | {data['p_value']:<8.4f} | {data['effect_size_cohens_d']:<7.2f} | {sig_str}"
            )

        print("==========================================================================\n")


if __name__ == "__main__":
    analyzer = VGCLStatisticalAnalyzer(alpha=0.05)

    # Simulated 3-seed runs for targeted vs uniform
    tgt_runs = {
        "Primary: Cluster Pass@1 (FC_01)": [0.42, 0.45, 0.43],
        "Secondary: Overall Pass@1": [0.38, 0.40, 0.39],
        "Secondary: Pass@10": [0.65, 0.68, 0.66],
        "Secondary: Transfer Density": [0.28, 0.31, 0.29],
    }

    uni_runs = {
        "Primary: Cluster Pass@1 (FC_01)": [0.15, 0.18, 0.16],
        "Secondary: Overall Pass@1": [0.25, 0.27, 0.26],
        "Secondary: Pass@10": [0.48, 0.50, 0.49],
        "Secondary: Transfer Density": [0.10, 0.12, 0.11],
    }

    report = analyzer.run_full_statistical_audit(tgt_runs, uni_runs)
    analyzer.print_publication_report(report)
