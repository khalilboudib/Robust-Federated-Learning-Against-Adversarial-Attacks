"""Plot Week 3 adversarial federated learning results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_LABELS = {
    "A_FedAvg_No_Attack": "FedAvg, no attack",
    "B_FedAvg_LabelFlip": "FedAvg, label flip",
    "C_Krum_LabelFlip": "Krum, label flip",
    "D_Median_LabelFlip": "Median, label flip",
    "E_TrimmedMean_LabelFlip": "Trimmed mean, label flip",
}


def load_results(results_path: Path) -> pd.DataFrame:
    results = pd.read_csv(results_path)
    results["scenario_label"] = results["scenario"].map(SCENARIO_LABELS).fillna(results["scenario"])
    return results


def plot_accuracy_curves(results: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(10, 5.5))
    ax = sns.lineplot(
        data=results,
        x="round",
        y="test_accuracy",
        hue="scenario_label",
        marker="o",
        linewidth=2,
    )
    ax.set_title("Week 3 FL Robustness: Test Accuracy by Communication Round")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy")
    y_min = max(0.0, results["test_accuracy"].min() - 0.03)
    y_max = min(1.0, results["test_accuracy"].max() + 0.03)
    ax.set_ylim(y_min, y_max)
    ax.legend(title="Scenario", loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_aggregation_time(results: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    summary = (
        results.groupby("aggregation", as_index=False)["aggregation_time_sec"]
        .mean()
        .sort_values("aggregation_time_sec", ascending=False)
    )
    label_map = {
        "fedavg": "FedAvg",
        "krum": "Krum",
        "coordinate_median": "Median",
        "trimmed_mean": "Trimmed mean",
    }
    summary["aggregation_label"] = summary["aggregation"].map(label_map).fillna(summary["aggregation"])

    plt.figure(figsize=(7.5, 4.5))
    ax = sns.barplot(
        data=summary,
        x="aggregation_label",
        y="aggregation_time_sec",
        palette=["#4C78A8", "#F58518", "#54A24B", "#B279A2"],
    )
    ax.set_title("Average Aggregation Computation Time")
    ax.set_xlabel("Aggregation method")
    ax.set_ylabel("Mean time per round (seconds)")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Week 3 result plots")
    parser.add_argument("--results", type=Path, default=ROOT / "experiments" / "week3_results.csv")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.results)
    accuracy_path = args.figures_dir / "week3_accuracy_curves.png"
    time_path = args.figures_dir / "week3_aggregation_time.png"
    plot_accuracy_curves(results, accuracy_path)
    plot_aggregation_time(results, time_path)
    print(f"Saved accuracy curves to {accuracy_path}")
    print(f"Saved aggregation timing chart to {time_path}")


if __name__ == "__main__":
    main()
