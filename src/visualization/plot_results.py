"""
Generate comparison plots from experiment results.

Reads outputs/metrics/results.csv and creates visualizations for the report.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


def load_results(csv_path: str = "outputs/metrics/results.csv") -> pd.DataFrame:
    """Load experiment results from CSV."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Results not found: {csv_path}")
    return pd.read_csv(csv_path)


def plot_f2_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot F2 score comparison across experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by experiment and average across folds
    grouped = df.groupby("experiment_id").agg(
        {"f2": "mean", "recall": "mean", "precision": "mean"}
    )
    grouped = grouped.sort_values("f2", ascending=False)

    x = range(len(grouped))
    ax.bar(x, grouped["f2"], color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=45, ha="right")
    ax.set_ylabel("F2 Score")
    ax.set_title("F2 Score Comparison Across Experiments")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(grouped["f2"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "f2_comparison.png")
    print("✓ Saved: f2_comparison.png")
    plt.close()


def plot_precision_recall(df: pd.DataFrame, output_dir: Path):
    """Plot precision vs recall scatter."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for exp_id in df["experiment_id"].unique():
        exp_data = df[df["experiment_id"] == exp_id]
        ax.scatter(
            exp_data["recall"],
            exp_data["precision"],
            label=exp_id,
            alpha=0.7,
            s=100,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall.png")
    print("✓ Saved: precision_recall.png")
    plt.close()


def plot_speed_vs_accuracy(df: pd.DataFrame, output_dir: Path):
    """Plot speed (ms/frame) vs F2 score."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_id in df["experiment_id"].unique():
        exp_data = df[df["experiment_id"] == exp_id]
        ax.scatter(
            exp_data["ms_per_frame"],
            exp_data["f2"],
            label=exp_id,
            alpha=0.7,
            s=100,
        )

    ax.set_xlabel("Latency (ms/frame)")
    ax.set_ylabel("F2 Score")
    ax.set_title("Speed vs Accuracy Trade-off")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "speed_vs_accuracy.png")
    print("✓ Saved: speed_vs_accuracy.png")
    plt.close()


def plot_metrics_table(df: pd.DataFrame, output_dir: Path):
    """Generate summary table image."""
    # Group by experiment and average
    summary = df.groupby("experiment_id").agg(
        {
            "f2": "mean",
            "recall": "mean",
            "precision": "mean",
            "ms_per_frame": "mean",
            "fold_id": "count",  # Number of runs
        }
    )
    summary.columns = ["F2", "Recall", "Precision", "ms/frame", "Runs"]
    summary = summary.sort_values("F2", ascending=False)

    fig, ax = plt.subplots(figsize=(10, len(summary) * 0.5 + 1))
    ax.axis("tight")
    ax.axis("off")

    # Format values
    table_data = summary.round(4).astype(str).values
    table = ax.table(
        cellText=table_data,
        rowLabels=summary.index,
        colLabels=summary.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_table.png")
    print("✓ Saved: metrics_table.png")
    plt.close()


def plot_inference_modes(df: pd.DataFrame, output_dir: Path):
    """Compare inference modes (standard, sahi, tracking)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by inference mode
    if "inference" in df.columns:
        grouped = df.groupby("inference")["f2"].mean().sort_values(ascending=False)
        x = range(len(grouped))
        ax.bar(x, grouped.values, color="coral", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index)
        ax.set_ylabel("F2 Score")
        ax.set_title("F2 Score by Inference Mode")
        ax.grid(axis="y", alpha=0.3)

        for i, v in enumerate(grouped.values):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(output_dir / "inference_modes.png")
        print("✓ Saved: inference_modes.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate experiment result plots")
    parser.add_argument(
        "--results",
        type=str,
        default="outputs/metrics/results.csv",
        help="Path to results CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/figures",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results}")
    df = load_results(args.results)
    print(f"Found {len(df)} experiment runs")

    if len(df) == 0:
        print("No results to plot. Run experiments first!")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print(f"\nGenerating plots in {output_dir}/")
    plot_f2_comparison(df, output_dir)
    plot_precision_recall(df, output_dir)
    plot_speed_vs_accuracy(df, output_dir)
    plot_metrics_table(df, output_dir)
    plot_inference_modes(df, output_dir)

    print(f"\n✓ All plots saved to {output_dir}/")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
