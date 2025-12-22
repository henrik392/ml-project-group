"""
Compare results from ablation study experiments.

Generates comparison tables and visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def load_experiment_results(base_dir: Path, fold: int) -> pd.DataFrame:
    """Load results from all ablation experiments."""
    results = []

    for exp_dir in sorted(base_dir.glob(f"*_fold{fold}")):
        results_csv = exp_dir / "results.csv"
        args_yaml = exp_dir / "args.yaml"

        if not results_csv.exists():
            print(f"⚠️  No results.csv found in {exp_dir.name}")
            continue

        # Load final epoch metrics
        df = pd.read_csv(results_csv)
        if df.empty:
            continue

        final_row = df.iloc[-1]

        # Parse experiment name
        exp_name = exp_dir.name.replace(f"_fold{fold}", "")

        results.append({
            "Experiment": exp_name,
            "mAP50": final_row.get("metrics/mAP50(B)", 0.0),
            "mAP50-95": final_row.get("metrics/mAP50-95(B)", 0.0),
            "Precision": final_row.get("metrics/precision(B)", 0.0),
            "Recall": final_row.get("metrics/recall(B)", 0.0),
            "Box Loss (train)": final_row.get("train/box_loss", 0.0),
            "Cls Loss (train)": final_row.get("train/cls_loss", 0.0),
            "Box Loss (val)": final_row.get("val/box_loss", 0.0),
            "Cls Loss (val)": final_row.get("val/cls_loss", 0.0),
        })

    return pd.DataFrame(results)


def print_comparison_table(df: pd.DataFrame):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("ABLATION STUDY RESULTS - PARAMETER ISOLATION")
    print("="*100 + "\n")

    # Sort by mAP50
    df_sorted = df.sort_values("mAP50", ascending=False).reset_index(drop=True)

    print("Detection Metrics (sorted by mAP50):\n")
    print(df_sorted[["Experiment", "mAP50", "mAP50-95", "Precision", "Recall"]].to_string(index=False))

    print("\n" + "-"*100 + "\n")

    print("Training Losses:\n")
    print(df_sorted[["Experiment", "Box Loss (train)", "Cls Loss (train)", "Box Loss (val)", "Cls Loss (val)"]].to_string(index=False))

    print("\n" + "="*100)

    # Highlight best and worst
    best = df_sorted.iloc[0]
    worst = df_sorted.iloc[-1]

    print(f"\n✅ BEST: {best['Experiment']}")
    print(f"   mAP50: {best['mAP50']:.4f} | Recall: {best['Recall']:.4f}")

    print(f"\n⚠️  WORST: {worst['Experiment']}")
    print(f"   mAP50: {worst['mAP50']:.4f} | Recall: {worst['Recall']:.4f}")

    # Compare to baseline if it exists
    baseline = df_sorted[df_sorted["Experiment"] == "baseline_control"]
    if not baseline.empty:
        print(f"\n📊 Baseline mAP50: {baseline.iloc[0]['mAP50']:.4f}")

        print("\nRelative to Baseline:")
        for _, row in df_sorted.iterrows():
            if row["Experiment"] == "baseline_control":
                continue
            diff = row["mAP50"] - baseline.iloc[0]["mAP50"]
            pct = (diff / baseline.iloc[0]["mAP50"]) * 100
            symbol = "✅" if diff > 0 else "⚠️"
            print(f"  {symbol} {row['Experiment']}: {diff:+.4f} ({pct:+.1f}%)")

    print()


def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Create comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by mAP50
    df_sorted = df.sort_values("mAP50", ascending=False)

    # 1. Detection metrics comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    x = range(len(df_sorted))
    width = 0.2

    ax.bar([i - 1.5*width for i in x], df_sorted["mAP50"],
           width, label="mAP50", alpha=0.8)
    ax.bar([i - 0.5*width for i in x], df_sorted["mAP50-95"],
           width, label="mAP50-95", alpha=0.8)
    ax.bar([i + 0.5*width for i in x], df_sorted["Precision"],
           width, label="Precision", alpha=0.8)
    ax.bar([i + 1.5*width for i in x], df_sorted["Recall"],
           width, label="Recall", alpha=0.8)

    ax.set_xlabel("Experiment", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Ablation Study: Detection Metrics Comparison", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["Experiment"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "ablation_metrics_comparison.png", dpi=300, bbox_inches="tight")
    print(f"✅ Saved: ablation_metrics_comparison.png")
    plt.close()

    # 2. mAP50 ranking
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#2ecc71' if exp == 'baseline_control' else '#3498db'
              for exp in df_sorted["Experiment"]]

    bars = ax.barh(df_sorted["Experiment"], df_sorted["mAP50"], color=colors, alpha=0.8)

    ax.set_xlabel("mAP50", fontweight="bold")
    ax.set_title("Ablation Study: mAP50 Ranking", fontweight="bold", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_sorted["mAP50"])):
        ax.text(val, bar.get_y() + bar.get_height()/2,
                f" {val:.4f}",
                va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "ablation_map50_ranking.png", dpi=300, bbox_inches="tight")
    print(f"✅ Saved: ablation_map50_ranking.png")
    plt.close()

    # 3. Relative to baseline (if exists)
    baseline = df_sorted[df_sorted["Experiment"] == "baseline_control"]
    if not baseline.empty:
        baseline_map50 = baseline.iloc[0]["mAP50"]

        df_relative = df_sorted[df_sorted["Experiment"] != "baseline_control"].copy()
        df_relative["Relative_mAP50"] = (df_relative["mAP50"] - baseline_map50) / baseline_map50 * 100

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_relative["Relative_mAP50"]]
        bars = ax.barh(df_relative["Experiment"], df_relative["Relative_mAP50"],
                      color=colors, alpha=0.8)

        ax.set_xlabel("Change vs Baseline (%)", fontweight="bold")
        ax.set_title("Ablation Study: Performance vs Baseline", fontweight="bold", fontsize=14)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, df_relative["Relative_mAP50"]):
            ax.text(val, bar.get_y() + bar.get_height()/2,
                    f" {val:+.1f}%",
                    va="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_dir / "ablation_vs_baseline.png", dpi=300, bbox_inches="tight")
        print(f"✅ Saved: ablation_vs_baseline.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare ablation study results")
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="runs/ablation",
        help="Base directory for experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/ablation_study",
        help="Output directory for comparison figures",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)

    if not base_dir.exists():
        print(f"❌ Error: {base_dir} does not exist")
        return

    # Load results
    print("Loading experiment results...")
    df = load_experiment_results(base_dir, args.fold)

    if df.empty:
        print("❌ No results found!")
        return

    print(f"✅ Loaded {len(df)} experiments\n")

    # Print comparison
    print_comparison_table(df)

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_comparison(df, output_dir)

    # Save CSV
    csv_path = output_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: ablation_results.csv")

    print(f"\n{'='*100}")
    print(f"✅ Comparison complete! Results saved to: {output_dir}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
