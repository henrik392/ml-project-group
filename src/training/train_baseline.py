"""
Baseline YOLOv11 training script for COTS detection.

Phase 1: Simple baseline with default hyperparameters.
"""

import argparse
from pathlib import Path
import torch

from ultralytics import YOLO


def train_fold(
    fold: int = 0,
    model_size: str = "n",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    cache: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    project: str = "runs/train",
) -> None:
    """
    Train YOLOv11 on a specific fold.

    Args:
        fold: Fold number (0, 1, or 2)
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        cache: Cache images ('ram', 'disk', or None)
        device: Device to use ('mps' for M4 Max, 'cuda' for GPU, 'cpu')
        project: Project directory for saving runs
    """
    print(f"\n{'='*80}")
    print(f"Training YOLOv11{model_size} on Fold {fold}")
    print(f"{'='*80}\n")

    # Load model
    model = YOLO(f"yolo11{model_size}.pt")  # Load pretrained model

    # Dataset config
    data_config = f"configs/dataset_fold_{fold}.yaml"

    # Verify config exists
    if not Path(data_config).exists():
        raise FileNotFoundError(f"Dataset config not found: {data_config}")

    # Train
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        cache=cache,
        device=device,
        project=project,
        name=f"yolo11{model_size}_fold{fold}",
        # Baseline settings
        patience=10,  # Early stopping patience
        save=True,
        plots=True,
        val=True,
        # M4 Max optimizations
        amp=True,  # Automatic Mixed Precision
        verbose=True,
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
    cache: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Train on all 3 folds for cross-validation.

    Args:
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size
        cache: Cache images
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
            cache=cache,
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

    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train YOLOv11 baseline")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold to train (0, 1, or 2). If None, train all folds.")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="Model size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--cache", type=str, default=None, choices=["ram", "disk"], help="Cache images (ram, disk)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (mps, cuda, cpu)")

    args = parser.parse_args()

    if args.fold is not None:
        # Train single fold
        train_fold(
            fold=args.fold,
            model_size=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            cache=args.cache,
            device=args.device,
        )
    else:
        # Train all folds
        train_all_folds(
            model_size=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            cache=args.cache,
            device=args.device,
        )


if __name__ == "__main__":
    main()
