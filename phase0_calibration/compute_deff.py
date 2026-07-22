import json
import numpy as np
from pathlib import Path


def main():
    results_file = Path("data/trajectory_results.json")
    if not results_file.exists():
        raise FileNotFoundError(
            "data/trajectory_results.json not found. Wait for extract_trajectories.py to finish.")

    with open(results_file, "r") as f:
        data = json.load(f)

    kmeans_labels = np.array(data["kmeans_labels"])
    human_labels = np.array(data["human_labels"])
    ari = data["ari"]

    # Use mechanistic clusters if ARI is below threshold, otherwise use human proxies
    labels = kmeans_labels if ari < 0.3 else human_labels

    unique_cats, counts = np.unique(labels, return_counts=True)
    num_categories = len(unique_cats)

    m_bar = np.mean(counts)  # Average cluster size

    # Baseline intra-cluster correlation estimate for LLM code outputs
    rho = 0.20

    # Unequal cluster size DEFF calculation
    deff = 1 + (m_bar - 1) * rho
    total_n = len(labels)
    n_eff = total_n / deff

    print("\n" + "=" * 60)
    print("  PHASE 0.2: DESIGN EFFECT (DEFF) & N_EFF CALCULATION")
    print("=" * 60)
    print(f"Total Calibration Samples (n): {total_n}")
    print(f"Active Taxonomy Clusters:     {num_categories}")
    print(f"Average Category Size (m_bar): {m_bar:.2f}")
    print(f"Assumed Intra-Cluster Rho:    {rho:.2f}")
    print(f"Design Effect (DEFF):         {deff:.3f}")
    print(f"Effective Sample Size (n_eff): {n_eff:.1f}")
    print("=" * 60)

    if deff > 1.5:
        print(f"\nWARNING: High cluster correlation (DEFF = {deff:.2f}).")
        print(
            f"Your dataset of {total_n} samples only acts like {int(n_eff)} independent samples.")
        print(
            f"In Phase 1, scale problem generation targets up by a factor of {deff:.2f}x.")
    else:
        print("\nDEFF is within acceptable bounds. Phase 0 calibration complete.")


if __name__ == "__main__":
    main()
