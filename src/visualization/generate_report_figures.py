"""
Generate publication-quality figures for final report and presentation.

This script creates comparison plots between baseline and optimized models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BASELINE_RESULTS = PROJECT_ROOT / "runs/train/yolo11n_fold02/results.csv"
PHASE2_RESULTS = PROJECT_ROOT / "runs/optimized/yolo11n_fold0_opt7/results.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports/final_report/figures"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load training results from CSV."""
    df = pd.read_csv(csv_path)
    df = df.dropna()  # Remove incomplete rows
    return df


def plot_performance_comparison():
    """Compare final metrics between baseline and Phase 2."""
    # Data
    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
    baseline = [0.154, 0.078, 0.620, 0.091]
    phase2 = [0.082, 0.040, 0.499, 0.052]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline (Phase 1)',
                   color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, phase2, width, label='Phase 2 (Optimized)',
                   color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Performance Comparison: Baseline vs Phase 2\n(YOLOv11n, Fold 0)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: performance_comparison.png")
    plt.close()


def plot_training_curves():
    """Plot training and validation losses over time."""
    baseline_df = load_results(BASELINE_RESULTS)
    phase2_df = load_results(PHASE2_RESULTS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress Comparison: Baseline vs Phase 2',
                 fontweight='bold', fontsize=16)

    # mAP50
    axes[0, 0].plot(baseline_df['epoch'], baseline_df['metrics/mAP50(B)'],
                    marker='o', label='Baseline', linewidth=2, markersize=4)
    axes[0, 0].plot(phase2_df['epoch'], phase2_df['metrics/mAP50(B)'],
                    marker='s', label='Phase 2', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('mAP50')
    axes[0, 0].set_title('mAP@0.5 Progression', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Recall
    axes[0, 1].plot(baseline_df['epoch'], baseline_df['metrics/recall(B)'],
                    marker='o', label='Baseline', linewidth=2, markersize=4)
    axes[0, 1].plot(phase2_df['epoch'], phase2_df['metrics/recall(B)'],
                    marker='s', label='Phase 2', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Recall Progression', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Training Box Loss
    axes[1, 0].plot(baseline_df['epoch'], baseline_df['train/box_loss'],
                    marker='o', label='Baseline', linewidth=2, markersize=4)
    axes[1, 0].plot(phase2_df['epoch'], phase2_df['train/box_loss'],
                    marker='s', label='Phase 2', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Box Loss')
    axes[1, 0].set_title('Training Box Loss', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Validation Classification Loss
    axes[1, 1].plot(baseline_df['epoch'], baseline_df['val/cls_loss'],
                    marker='o', label='Baseline', linewidth=2, markersize=4)
    axes[1, 1].plot(phase2_df['epoch'], phase2_df['val/cls_loss'],
                    marker='s', label='Phase 2', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Classification Loss')
    axes[1, 1].set_title('Validation Classification Loss', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: training_curves.png")
    plt.close()


def plot_phase2_progression():
    """Detailed Phase 2 training progression."""
    phase2_df = load_results(PHASE2_RESULTS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 2 Detailed Training Progression (15 Epochs)',
                 fontweight='bold', fontsize=16)

    # All metrics
    axes[0, 0].plot(phase2_df['epoch'], phase2_df['metrics/precision(B)'],
                    marker='o', label='Precision', linewidth=2)
    axes[0, 0].plot(phase2_df['epoch'], phase2_df['metrics/recall(B)'],
                    marker='s', label='Recall', linewidth=2)
    axes[0, 0].plot(phase2_df['epoch'], phase2_df['metrics/mAP50(B)'],
                    marker='^', label='mAP50', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Detection Metrics', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Training losses
    axes[0, 1].plot(phase2_df['epoch'], phase2_df['train/box_loss'],
                    marker='o', label='Box Loss', linewidth=2)
    axes[0, 1].plot(phase2_df['epoch'], phase2_df['train/cls_loss'],
                    marker='s', label='Classification Loss', linewidth=2)
    axes[0, 1].plot(phase2_df['epoch'], phase2_df['train/dfl_loss'],
                    marker='^', label='DFL Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Losses', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Validation losses
    axes[1, 0].plot(phase2_df['epoch'], phase2_df['val/box_loss'],
                    marker='o', label='Box Loss', linewidth=2)
    axes[1, 0].plot(phase2_df['epoch'], phase2_df['val/cls_loss'],
                    marker='s', label='Classification Loss', linewidth=2)
    axes[1, 0].plot(phase2_df['epoch'], phase2_df['val/dfl_loss'],
                    marker='^', label='DFL Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Validation Losses', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Training time per epoch
    epoch_times = phase2_df['time'].diff().fillna(phase2_df['time'].iloc[0])
    axes[1, 1].bar(phase2_df['epoch'], epoch_times / 60, alpha=0.7, color='#3498db')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (minutes)')
    axes[1, 1].set_title('Training Time per Epoch', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Normal (~10min)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phase2_progression.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: phase2_progression.png")
    plt.close()


def create_summary_table():
    """Create a summary table comparing all key metrics."""
    data = {
        'Metric': ['mAP50', 'mAP50-95', 'Precision', 'Recall',
                   'Box Loss (train)', 'Cls Loss (train)', 'DFL Loss (train)',
                   'Box Loss (val)', 'Cls Loss (val)', 'DFL Loss (val)'],
        'Baseline (Epoch 10)': [
            0.154, 0.078, 0.620, 0.091,
            1.756, 1.802, 1.005,
            1.990, 3.777, 1.023
        ],
        'Phase 2 (Epoch 15)': [
            0.082, 0.040, 0.499, 0.052,
            1.030, 0.929, 0.950,
            1.305, 5.139, 1.057
        ],
        'Change (%)': [
            -46.8, -48.7, -19.5, -42.9,
            -41.3, -48.4, -5.5,
            -34.4, +36.0, +3.3
        ]
    }

    df = pd.DataFrame(data)

    # Save to CSV
    csv_path = OUTPUT_DIR / 'performance_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: performance_summary.csv")

    # Create visual table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Color code the change column
    colors = []
    for val in df['Change (%)']:
        if val < 0 and 'Loss' in df.loc[df['Change (%)'] == val, 'Metric'].values[0]:
            colors.append('#90EE90')  # Light green for loss decrease
        elif val < 0:
            colors.append('#FFB6C1')  # Light red for metric decrease
        else:
            colors.append('#FFB6C1')  # Light red for loss increase

    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.3, 0.25, 0.25, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color change column
    for i in range(1, len(df) + 1):
        table[(i, 3)].set_facecolor(colors[i-1])

    plt.title('Performance Metrics Comparison: Baseline vs Phase 2\n',
             fontweight='bold', fontsize=14, pad=20)

    plt.savefig(OUTPUT_DIR / 'performance_table.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: performance_table.png")
    plt.close()


def main():
    """Generate all figures for the report."""
    print("\n" + "="*60)
    print("Generating Report Figures")
    print("="*60 + "\n")

    try:
        print("📊 Creating performance comparison...")
        plot_performance_comparison()

        print("📈 Creating training curves...")
        plot_training_curves()

        print("📉 Creating Phase 2 progression plots...")
        plot_phase2_progression()

        print("📋 Creating summary table...")
        create_summary_table()

        print("\n" + "="*60)
        print(f"✅ All figures saved to: {OUTPUT_DIR}")
        print("="*60 + "\n")

        print("Generated files:")
        for file in OUTPUT_DIR.glob('*.png'):
            print(f"  - {file.name}")
        for file in OUTPUT_DIR.glob('*.csv'):
            print(f"  - {file.name}")

    except Exception as e:
        print(f"\n❌ Error generating figures: {e}")
        raise


if __name__ == "__main__":
    main()
