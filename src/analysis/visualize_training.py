"""Generate publication-quality training visualizations for baseline model."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "serif"


def load_results(run_path: str) -> pd.DataFrame:
    """Load training results from CSV."""
    csv_path = Path(run_path) / "results.csv"
    df = pd.read_csv(csv_path)
    # Clean column names
    df.columns = df.columns.str.strip()
    return df


def plot_training_curves(df: pd.DataFrame, output_dir: Path, run_name: str):
    """Generate comprehensive training curves."""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Progress: {run_name}", fontsize=14, fontweight="bold")

    # 1. Loss curves (box, cls, dfl)
    ax = axes[0, 0]
    ax.plot(
        df["epoch"], df["train/box_loss"], "b-", label="Train Box Loss", linewidth=2
    )
    ax.plot(
        df["epoch"],
        df["val/box_loss"],
        "b--",
        label="Val Box Loss",
        linewidth=2,
        alpha=0.7,
    )
    ax.plot(
        df["epoch"], df["train/cls_loss"], "r-", label="Train Cls Loss", linewidth=2
    )
    ax.plot(
        df["epoch"],
        df["val/cls_loss"],
        "r--",
        label="Val Cls Loss",
        linewidth=2,
        alpha=0.7,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Losses")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Precision and Recall
    ax = axes[0, 1]
    ax.plot(
        df["epoch"],
        df["metrics/precision(B)"],
        "g-",
        label="Precision",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    ax.plot(
        df["epoch"],
        df["metrics/recall(B)"],
        "orange",
        label="Recall",
        linewidth=2,
        marker="s",
        markersize=3,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Precision and Recall")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 3. mAP scores
    ax = axes[1, 0]
    ax.plot(
        df["epoch"],
        df["metrics/mAP50(B)"],
        "purple",
        label="mAP@0.5",
        linewidth=2,
        marker="D",
        markersize=3,
    )
    ax.plot(
        df["epoch"],
        df["metrics/mAP50-95(B)"],
        "brown",
        label="mAP@0.5:0.95",
        linewidth=2,
        marker="^",
        markersize=3,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP Score")
    ax.set_title("Mean Average Precision")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(df["metrics/mAP50(B)"].max() * 1.2, 0.2)])

    # 4. Learning rate
    ax = axes[1, 1]
    ax.plot(df["epoch"], df["lr/pg0"], "k-", label="Learning Rate", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    output_path = output_dir / f"{run_name}_training_curves.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved training curves to: {output_path}")
    plt.close()


def plot_metrics_summary(df: pd.DataFrame, output_dir: Path, run_name: str):
    """Generate summary metrics comparison."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get final metrics
    final_metrics = {
        "Precision": df["metrics/precision(B)"].iloc[-1],
        "Recall": df["metrics/recall(B)"].iloc[-1],
        "mAP@0.5": df["metrics/mAP50(B)"].iloc[-1],
        "mAP@0.5:0.95": df["metrics/mAP50-95(B)"].iloc[-1],
    }

    # Create bar plot
    colors = ["#2ecc71", "#e74c3c", "#9b59b6", "#e67e22"]
    bars = ax.bar(
        final_metrics.keys(),
        final_metrics.values(),
        color=colors,
        alpha=0.8,
        edgecolor="black",
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Final Validation Metrics - {run_name}", fontsize=14, fontweight="bold"
    )
    ax.set_ylim([0, max(final_metrics.values()) * 1.3])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / f"{run_name}_metrics_summary.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved metrics summary to: {output_path}")
    plt.close()


def plot_loss_components(df: pd.DataFrame, output_dir: Path, run_name: str):
    """Plot detailed loss component analysis."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Loss Component Analysis: {run_name}", fontsize=14, fontweight="bold")

    loss_types = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/cls_loss", "val/cls_loss", "Classification Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
    ]

    for ax, (train_col, val_col, title) in zip(axes, loss_types):
        ax.plot(df["epoch"], df[train_col], "b-", label="Train", linewidth=2)
        ax.plot(
            df["epoch"], df[val_col], "r--", label="Validation", linewidth=2, alpha=0.7
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"{run_name}_loss_components.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved loss components to: {output_path}")
    plt.close()


def generate_statistics_table(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the training run."""

    final_epoch = df.iloc[-1]
    best_map_epoch = df.loc[df["metrics/mAP50(B)"].idxmax()]

    stats = {
        "total_epochs": len(df),
        "final_metrics": {
            "precision": final_epoch["metrics/precision(B)"],
            "recall": final_epoch["metrics/recall(B)"],
            "mAP50": final_epoch["metrics/mAP50(B)"],
            "mAP50_95": final_epoch["metrics/mAP50-95(B)"],
        },
        "best_epoch": {
            "epoch": int(best_map_epoch["epoch"]),
            "mAP50": best_map_epoch["metrics/mAP50(B)"],
            "precision": best_map_epoch["metrics/precision(B)"],
            "recall": best_map_epoch["metrics/recall(B)"],
        },
        "loss_improvement": {
            "box_loss": (
                (df["train/box_loss"].iloc[0] - df["train/box_loss"].iloc[-1])
                / df["train/box_loss"].iloc[0]
                * 100
            ),
            "cls_loss": (
                (df["train/cls_loss"].iloc[0] - df["train/cls_loss"].iloc[-1])
                / df["train/cls_loss"].iloc[0]
                * 100
            ),
            "dfl_loss": (
                (df["train/dfl_loss"].iloc[0] - df["train/dfl_loss"].iloc[-1])
                / df["train/dfl_loss"].iloc[0]
                * 100
            ),
        },
    }

    return stats


def main():
    """Generate all visualizations for baseline training."""

    # Configuration
    run_path = "runs/train/yolo11n_fold05"
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = "YOLOv11n_Fold0_30Epochs"

    print(f"Loading results from: {run_path}")
    df = load_results(run_path)

    print("\nGenerating visualizations...")
    plot_training_curves(df, output_dir, run_name)
    plot_metrics_summary(df, output_dir, run_name)
    plot_loss_components(df, output_dir, run_name)

    print("\nGenerating statistics...")
    stats = generate_statistics_table(df)

    print(f"\n{'=' * 80}")
    print(f"TRAINING SUMMARY: {run_name}")
    print(f"{'=' * 80}")
    print(f"\nTotal Epochs: {stats['total_epochs']}")
    print(f"\nFinal Metrics (Epoch {stats['total_epochs']}):")
    print(f"  Precision:    {stats['final_metrics']['precision']:.4f}")
    print(f"  Recall:       {stats['final_metrics']['recall']:.4f}")
    print(f"  mAP@0.5:      {stats['final_metrics']['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {stats['final_metrics']['mAP50_95']:.4f}")

    print(f"\nBest Performance (Epoch {stats['best_epoch']['epoch']}):")
    print(f"  mAP@0.5:      {stats['best_epoch']['mAP50']:.4f}")
    print(f"  Precision:    {stats['best_epoch']['precision']:.4f}")
    print(f"  Recall:       {stats['best_epoch']['recall']:.4f}")

    print("\nLoss Improvements (Train):")
    print(f"  Box Loss:     {stats['loss_improvement']['box_loss']:.1f}%")
    print(f"  Cls Loss:     {stats['loss_improvement']['cls_loss']:.1f}%")
    print(f"  DFL Loss:     {stats['loss_improvement']['dfl_loss']:.1f}%")
    print(f"{'=' * 80}\n")

    print("All visualizations generated successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
