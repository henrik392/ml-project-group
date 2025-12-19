"""
Optimized YOLOv11 training script for COTS detection.

Phase 2: Hyperparameter optimization and augmentations based on 1st place solution.

Key improvements from baseline:
- Modified hyperparameters: box=0.2, iou=0.3
- Augmentations: rotation, mixup, flips (NO HSV)
- 50 epochs for proper convergence
- Lower confidence threshold for higher recall
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train_fold(
    fold: int = 0,
    model_size: str = "n",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "mps",
    project: str = "runs/optimized",
) -> None:
    """
    Train YOLOv11 on a specific fold with optimized hyperparameters.

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
    print(f"Training YOLOv11{model_size} (Optimized) on Fold {fold}")
    print(f"{'='*80}\n")

    # Load model
    model = YOLO(f"yolo11{model_size}.pt")

    # Dataset config
    data_config = f"configs/dataset_fold_{fold}.yaml"

    # Verify config exists
    if not Path(data_config).exists():
        raise FileNotFoundError(f"Dataset config not found: {data_config}")

    # Train with optimized hyperparameters
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=f"yolo11{model_size}_fold{fold}_opt",
        # Training settings
        patience=15,  # Increased patience for 50 epochs
        save=True,
        plots=True,
        val=True,
        amp=True,  # Automatic Mixed Precision
        verbose=True,
        # Optimized hyperparameters (from 1st place solution)
        box=5.0,  # Box loss gain (default: 7.5) - reduced for small objects
        # NOTE: box=0.2 from winning solution may be too aggressive, causing TAL issues
        # NOTE: iou_t=0.3 is for inference/NMS, not training - set during prediction instead
        # Augmentations - minimal set to avoid TAL errors
        degrees=0.0,  # Disabled - rotation with mosaic causes TAL errors
        # mixup=0.1,  # Disabled - causing shape mismatch errors with task-aligned assigner
        fliplr=0.5,  # Horizontal flip probability (default: 0.5)
        flipud=0.0,  # No vertical flip (default: 0.0)
        # NO HSV augmentation (per winning solution)
        hsv_h=0.0,  # Hue augmentation (default: 0.015)
        hsv_s=0.0,  # Saturation augmentation (default: 0.7)
        hsv_v=0.0,  # Value augmentation (default: 0.4)
        # Other augmentations
        translate=0.1,  # Translation (default: 0.1)
        scale=0.5,  # Scale (default: 0.5)
        mosaic=1.0,  # Mosaic augmentation (default: 1.0)
        # Learning rate
        lr0=0.01,  # Initial learning rate (default: 0.01)
        lrf=0.01,  # Final learning rate factor (default: 0.01)
    )

    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Results saved to: {results.save_dir}")
    print(f"{'='*80}\n")

    # Print final metrics
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        print("Final Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    return results


def train_all_folds(
    model_size: str = "n",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "mps",
) -> dict:
    """
    Train on all 3 folds for cross-validation.

    Args:
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size
        device: Device to use

    Returns:
        Dict with results for each fold
    """
    results = {}

    for fold in range(3):
        print(f"\n{'#'*80}")
        print(f"# FOLD {fold}/3")
        print(f"{'#'*80}\n")

        fold_results = train_fold(
            fold=fold,
            model_size=model_size,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
        )

        results[f"fold_{fold}"] = fold_results

    # Summary
    print(f"\n{'='*80}")
    print("3-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}\n")

    for fold, fold_results in results.items():
        print(f"{fold}:")
        if hasattr(fold_results, "results_dict"):
            metrics = fold_results.results_dict
            print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
            print(f"  Recall: {metrics.get('metrics/recall(B)', 'N/A')}")

    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train YOLOv11 with optimized hyperparameters")
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specific fold to train (0, 1, or 2). If None, train all folds.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="Model size",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cuda, cpu)")

    args = parser.parse_args()

    if args.fold is not None:
        # Train single fold
        train_fold(
            fold=args.fold,
            model_size=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    else:
        # Train all folds
        train_all_folds(
            model_size=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )


if __name__ == "__main__":
    main()
