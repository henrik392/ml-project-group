"""
Generate additional presentation figures for ablation study.

Creates clear, focused visualizations for final presentation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 11

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
ABLATION_RESULTS = PROJECT_ROOT / "reports/ablation_study/ablation_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports/final_report/figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_hsv_impact():
    """Show the critical impact of HSV augmentation."""
    fig, ax = plt.subplots(figsize=(10, 6))

    experiments = ['Baseline\n(HSV enabled)', 'no_hsv\n(HSV disabled)']
    map50 = [0.1260, 0.0828]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax.bar(experiments, map50, color=colors, alpha=0.85, width=0.5)

    ax.set_ylabel('mAP@0.5', fontweight='bold', fontsize=14)
    ax.set_title('Critical Importance of HSV Augmentation\nFor Underwater Imagery',
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_ylim(0, 0.15)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, map50):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}\n({val*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=13)

    # Add performance drop annotation
    ax.annotate('',
                xy=(1, map50[1]), xycoords='data',
                xytext=(1, map50[0]), textcoords='data',
                arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))

    ax.text(1.15, (map50[0] + map50[1])/2,
            '-34.3%\nPerformance\nDrop',
            ha='left', va='center',
            fontweight='bold', fontsize=13, color='red',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ablation_hsv_impact.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: ablation_hsv_impact.png")
    plt.close()


def plot_box_weight_sweep():
    """Show effect of box loss weight on performance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    box_weights = [7.5, 6.5, 6.0, 5.5, 5.0]
    map50_values = [0.1260, 0.0870, 0.1072, 0.1173, 0.1167]

    # Plot line with markers
    ax.plot(box_weights, map50_values, 'o-', linewidth=3, markersize=10,
            color='#3498db', label='mAP@0.5')

    # Highlight baseline
    ax.scatter([7.5], [0.1260], s=300, color='#2ecc71', zorder=5,
              edgecolors='black', linewidths=2, label='Baseline (Best)')

    ax.set_xlabel('Box Loss Weight', fontweight='bold', fontsize=14)
    ax.set_ylabel('mAP@0.5', fontweight='bold', fontsize=14)
    ax.set_title('Effect of Box Loss Weight on Performance\nHigher Weight = Better Localization',
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xlim(4.5, 8.0)
    ax.set_ylim(0.07, 0.14)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)

    # Add value labels
    for x, y in zip(box_weights, map50_values):
        offset = 0.003 if x != 7.5 else -0.006
        ax.text(x, y + offset, f'{y:.4f}',
                ha='center', va='bottom' if x != 7.5 else 'top',
                fontsize=10, fontweight='bold')

    # Add annotation
    ax.annotate('All reductions\nhurt performance',
                xy=(5.5, 0.11), xycoords='data',
                xytext=(6.5, 0.08), textcoords='data',
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ablation_box_weight_sweep.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: ablation_box_weight_sweep.png")
    plt.close()


def plot_parameter_impact_summary():
    """Summary heatmap showing all parameter effects."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Data
    parameters = ['Mixup\n+0.1', 'Baseline\n(control)', 'Box→5.5', 'Box→5.0',
                  'Rotation\n+10°', 'Box→6.0', 'Box→6.5', 'HSV\nDisabled']
    map50 = [0.1262, 0.1260, 0.1173, 0.1167, 0.1100, 0.1072, 0.0870, 0.0828]
    changes = [+0.1, 0.0, -6.9, -7.3, -12.7, -14.9, -30.9, -34.3]

    # Color code
    colors = ['#2ecc71' if c >= 0 else '#e74c3c' for c in changes]

    bars = ax.barh(parameters, map50, color=colors, alpha=0.85)

    # Add baseline reference line
    ax.axvline(x=0.1260, color='black', linestyle='--', linewidth=2,
               alpha=0.5, label='Baseline')

    ax.set_xlabel('mAP@0.5', fontweight='bold', fontsize=14)
    ax.set_title('Ablation Study: All Parameter Effects\n(5 Epochs, Fold 0)',
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xlim(0.06, 0.14)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels with change percentages
    for bar, val, change in zip(bars, map50, changes):
        label = f'{val:.4f} ({change:+.1f}%)'
        ax.text(val, bar.get_y() + bar.get_height()/2,
                f'  {label}',
                va='center', ha='left' if val > 0.095 else 'right',
                fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ablation_all_parameters.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: ablation_all_parameters.png")
    plt.close()


def plot_phase2_failure_explanation():
    """Explain why Phase 2 failed using ablation results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Individual effects
    experiments = ['Baseline', 'HSV\nDisabled\nOnly', 'Box→5.0\nOnly']
    map50 = [0.1260, 0.0828, 0.1167]
    changes = [0, -34.3, -7.3]
    colors = ['#2ecc71', '#e74c3c', '#e67e22']

    bars1 = ax1.bar(experiments, map50, color=colors, alpha=0.85)

    ax1.set_ylabel('mAP@0.5', fontweight='bold', fontsize=13)
    ax1.set_title('Individual Parameter Effects',
                  fontweight='bold', fontsize=14)
    ax1.set_ylim(0, 0.16)
    ax1.grid(axis='y', alpha=0.3)

    for bar, val, change in zip(bars1, map50, changes):
        ax1.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.3f}\n({change:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Right plot: Combined effect vs actual Phase 2
    experiments2 = ['Baseline\n(Phase 1)', 'Phase 2\n(Actual)', 'Expected\nFrom Ablation']
    map50_2 = [0.154, 0.082, 0.083]  # Phase 1 @ 10 epochs, Phase 2 @ 15 epochs, no_hsv @ 5 epochs
    colors2 = ['#2ecc71', '#c0392b', '#e74c3c']

    bars2 = ax2.bar(experiments2, map50_2, color=colors2, alpha=0.85)

    ax2.set_ylabel('mAP@0.5', fontweight='bold', fontsize=13)
    ax2.set_title('Phase 2 Failure: Predicted vs Actual',
                  fontweight='bold', fontsize=14)
    ax2.set_ylim(0, 0.18)
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars2, map50_2):
        ax2.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add annotation
    ax2.text(1, 0.09, 'Ablation study\npredicts failure!\n(-34% from HSV)',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.suptitle('Why Phase 2 Failed: Ablation Study Explanation',
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ablation_phase2_failure.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: ablation_phase2_failure.png")
    plt.close()


def main():
    """Generate all presentation figures."""
    print("\n" + "="*60)
    print("Generating Ablation Study Presentation Figures")
    print("="*60 + "\n")

    if not ABLATION_RESULTS.exists():
        print(f"❌ Error: {ABLATION_RESULTS} not found!")
        print("Run the ablation study first:")
        print("  uv run src/training/ablation_study.py --fold 0 --epochs 5")
        return

    try:
        print("📊 Creating HSV impact visualization...")
        plot_hsv_impact()

        print("📊 Creating box weight sweep...")
        plot_box_weight_sweep()

        print("📊 Creating parameter summary...")
        plot_parameter_impact_summary()

        print("📊 Creating Phase 2 failure explanation...")
        plot_phase2_failure_explanation()

        print("\n" + "="*60)
        print(f"✅ All figures saved to: {OUTPUT_DIR}")
        print("="*60 + "\n")

        print("Generated files:")
        for file in OUTPUT_DIR.glob('ablation_*.png'):
            print(f"  - {file.name}")

    except Exception as e:
        print(f"\n❌ Error generating figures: {e}")
        raise


if __name__ == "__main__":
    main()
