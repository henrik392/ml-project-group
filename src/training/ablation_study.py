"""
Ablation Study: Test individual parameter changes from baseline.

Run multiple training configs in parallel to isolate parameter effects.
Each experiment changes ONE parameter from baseline.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Baseline configuration (from Phase 1)
BASELINE_PARAMS = {
    "box": 7.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "mixup": 0.0,
}

# Ablation experiments - each changes ONE parameter
EXPERIMENTS = [
    {
        "name": "baseline_control",
        "description": "Baseline (control)",
        **BASELINE_PARAMS,
    },
    {
        "name": "box_6.5",
        "description": "Box loss 7.5→6.5",
        **BASELINE_PARAMS,
        "box": 6.5,
    },
    {
        "name": "box_6.0",
        "description": "Box loss 7.5→6.0",
        **BASELINE_PARAMS,
        "box": 6.0,
    },
    {
        "name": "box_5.5",
        "description": "Box loss 7.5→5.5",
        **BASELINE_PARAMS,
        "box": 5.5,
    },
    {
        "name": "box_5.0",
        "description": "Box loss 7.5→5.0",
        **BASELINE_PARAMS,
        "box": 5.0,
    },
    {
        "name": "no_hsv",
        "description": "HSV disabled",
        **BASELINE_PARAMS,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    },
    {
        "name": "rotation_10",
        "description": "Rotation enabled (10°)",
        **BASELINE_PARAMS,
        "degrees": 10.0,
    },
    {
        "name": "mixup_0.1",
        "description": "Mixup enabled (0.1)",
        **BASELINE_PARAMS,
        "mixup": 0.1,
    },
]


def run_experiment(exp: dict, fold: int, epochs: int, device: str, dry_run: bool = False):
    """
    Run a single ablation experiment.

    Args:
        exp: Experiment configuration dict
        fold: Fold number (0, 1, or 2)
        epochs: Number of training epochs
        device: Device to use (mps, cuda, cpu)
        dry_run: If True, print command without running
    """
    cmd = [
        "uv", "run",
        "src/training/train_single_ablation.py",
        "--name", exp["name"],
        "--description", exp["description"],
        "--fold", str(fold),
        "--epochs", str(epochs),
        "--device", device,
        "--box", str(exp["box"]),
        "--hsv_h", str(exp["hsv_h"]),
        "--hsv_s", str(exp["hsv_s"]),
        "--hsv_v", str(exp["hsv_v"]),
        "--degrees", str(exp["degrees"]),
        "--mixup", str(exp["mixup"]),
    ]

    if dry_run:
        print(f"[DRY RUN] {exp['name']}: {exp['description']}")
        print(f"  Command: {' '.join(cmd)}")
        return None

    print(f"🚀 Starting: {exp['name']} - {exp['description']}")

    # Run in background
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return {
        "name": exp["name"],
        "description": exp["description"],
        "process": process,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold to train on (0, 1, or 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs (default: 5 for fast iteration)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device (mps, cuda, cpu)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Specific experiments to run (by name). If not provided, runs all.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run experiments sequentially instead of parallel",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment configs without running",
    )

    args = parser.parse_args()

    # Filter experiments
    if args.experiments:
        experiments = [e for e in EXPERIMENTS if e["name"] in args.experiments]
        if not experiments:
            print(f"❌ No experiments found matching: {args.experiments}")
            print(f"Available: {[e['name'] for e in EXPERIMENTS]}")
            return
    else:
        experiments = EXPERIMENTS

    print("\n" + "="*80)
    print("ABLATION STUDY: Isolating Parameter Effects")
    print("="*80)
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Experiments: {len(experiments)}")
    print(f"Mode: {'Sequential' if args.sequential else 'Parallel'}")
    print("="*80 + "\n")

    if args.dry_run:
        print("DRY RUN MODE - No training will be executed\n")

    # List experiments
    print("Experiments to run:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}: {exp['description']}")
    print()

    if args.dry_run:
        for exp in experiments:
            run_experiment(exp, args.fold, args.epochs, args.device, dry_run=True)
        return

    # Run experiments
    if args.sequential:
        # Sequential mode
        for exp in experiments:
            result = run_experiment(exp, args.fold, args.epochs, args.device)
            if result:
                print(f"⏳ Waiting for {result['name']} to complete...")
                stdout, stderr = result["process"].communicate()
                print(stdout)
                if stderr:
                    print(f"⚠️  Stderr: {stderr}")
    else:
        # Parallel mode
        processes = []
        for exp in experiments:
            result = run_experiment(exp, args.fold, args.epochs, args.device)
            if result:
                processes.append(result)
                time.sleep(2)  # Stagger starts slightly

        print(f"\n{'='*80}")
        print(f"✅ Launched {len(processes)} experiments in parallel")
        print(f"{'='*80}\n")

        print("Monitoring progress...")
        print("You can check individual runs with:")
        print("  ls -la runs/ablation/")
        print("  tail -f runs/ablation/<experiment_name>/train.log")
        print()

        # Wait for all to complete
        print("Waiting for all experiments to complete...")
        for i, p in enumerate(processes, 1):
            print(f"  [{i}/{len(processes)}] Waiting for {p['name']}...")
            stdout, stderr = p["process"].communicate()

            # Save output
            output_dir = Path(f"runs/ablation/{p['name']}_fold{args.fold}")
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / "stdout.log", "w") as f:
                f.write(stdout)

            if stderr:
                with open(output_dir / "stderr.log", "w") as f:
                    f.write(stderr)

        print(f"\n{'='*80}")
        print("✅ All experiments completed!")
        print(f"{'='*80}\n")
        print("Results saved to: runs/ablation/")


if __name__ == "__main__":
    main()
