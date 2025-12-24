"""
Phase 2 Revised: Conservative optimization for COTS detection.

Key changes from baseline:
1. KEEP HSV augmentation (critical for underwater imagery)
2. DISABLE rotation (avoid MPS TAL errors)
3. Small box weight reduction: 7.5 → 6.5 (conservative step)
4. Shorter training: 20 epochs (test stability)

Strategy: Incremental changes to isolate what works on MPS backend.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train_fold_revised(
    fold: int = 0,
    model_size: str = "n",
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "mps",
    project: str = "runs/phase2_revised",
) -> None:
    """
    Train YOLOv11 with revised Phase 2 settings.

    Args:
        fold: Fold number (0, 1, or 2)
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: Device to use ('mps' for M4 Max, 'cuda' for GPU, 'cpu')
        project: Project directory for saving runs
    """
    print(f"\n{'='*80}")
    print(f"Phase 2 Revised: Training YOLOv11{model_size} on Fold {fold}")
    print(f"{'='*80}\n")

    print("Configuration Changes:")
    print("  ✅ HSV augmentation: ENABLED (kept from baseline)")
    print("  ⚠️  Rotation: DISABLED (avoid TAL errors on MPS)")
    print("  📊 Box loss weight: 7.5 → 6.5 (conservative reduction)")
    print("  ⏱️  Epochs: 20 (stability test)")
    print()

    # Load model
    model = YOLO(f"yolo11{model_size}.pt")

    # Dataset config
    data_config = f"configs/dataset_fold_{fold}.yaml"

    # Verify config exists
    if not Path(data_config).exists():
        raise FileNotFoundError(f"Dataset config not found: {data_config}")

    # Train with revised settings
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=f"yolo11{model_size}_fold{fold}_rev1",
        # Training settings
        patience=15,
        save=True,
        plots=True,
        val=True,
        # M4 Max optimizations
        amp=True,
        verbose=True,
        # Hyperparameters - CONSERVATIVE CHANGES
        box=6.5,  # Reduced from 7.5 (baseline) - small step
        cls=0.5,  # Default
        dfl=1.5,  # Default
        # Augmentations - KEEP HSV, DISABLE ROTATION
        hsv_h=0.015,  # ENABLED (baseline setting)
        hsv_s=0.7,    # ENABLED (baseline setting)
        hsv_v=0.4,    # ENABLED (baseline setting)
        degrees=0.0,  # DISABLED (avoid TAL errors on MPS)
        mixup=0.0,    # Keep disabled (TAL issues)
        fliplr=0.5,   # Horizontal flips enabled
        mosaic=1.0,   # Standard mosaic
        translate=0.1,
        scale=0.5,
    )

    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Results saved to: {results.save_dir}")
    print(f"{'='*80}\n")

    # Print final metrics
    print("Final Metrics:")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"  Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")

    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train YOLOv11 Phase 2 Revised")
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold to train (0, 1, or 2)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="Model size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device (mps, cuda, cpu)",
    )

    args = parser.parse_args()

    train_fold_revised(
        fold=args.fold,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
