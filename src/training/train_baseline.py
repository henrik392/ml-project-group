"""
Baseline YOLOv11 training script for COTS detection.

Phase 1: Simple baseline with default hyperparameters.
"""

import argparse

from src.training.config import BaselineConfig
from src.training.utils import (
    find_latest_checkpoint,
    load_model_with_checkpoint,
    print_training_header,
    print_training_summary,
    verify_dataset_config,
    with_retry,
)


@with_retry(max_retries=5, wait_time=10)
def train_fold(
    fold: int = 0,
    model_size: str = "n",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "mps",
    project: str = "runs/train",
    resume: bool = False,
    _retry_count: int = 0,
) -> None:
    """
    Train YOLOv11 on a specific fold with automatic retry on errors.

    Args:
        fold: Fold number (0, 1, or 2)
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: Device to use ('mps' for M4 Max, 'cuda' for GPU, 'cpu')
        project: Project directory for saving runs
        resume: Resume from last checkpoint if available
        _retry_count: Internal retry counter (set by decorator)
    """
    # Create configuration
    config = BaselineConfig(
        fold=fold,
        model_size=model_size,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        resume=resume,
    )

    # Print training header
    print_training_header(
        fold=fold,
        model_size=model_size,
        phase="Baseline",
        auto_retry=True,
        max_retries=5,
    )

    # Verify dataset config exists
    verify_dataset_config(fold)

    # Find checkpoint if resuming or retrying
    checkpoint_path = None
    if resume or _retry_count > 0:
        checkpoint_path = find_latest_checkpoint(project, config.get_run_name())
        if checkpoint_path:
            if _retry_count > 0:
                print(f"\n[RETRY {_retry_count}/5] Resuming from checkpoint: {checkpoint_path}")
            else:
                print(f"Resuming from checkpoint: {checkpoint_path}")
        elif resume:
            print("No checkpoint found, starting from pretrained weights")

    # Load model
    model = load_model_with_checkpoint(model_size, checkpoint_path)

    # Get training kwargs
    train_kwargs = config.to_train_kwargs()

    # Set resume flag if we have a checkpoint
    if checkpoint_path:
        train_kwargs["resume"] = True

    # Train
    results = model.train(**train_kwargs)

    # Print summary
    print_training_summary(results, results.save_dir)

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
        print(f"\n{'#' * 80}")
        print(f"# FOLD {fold}/3")
        print(f"{'#' * 80}\n")

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
    print(f"\n{'=' * 80}")
    print("3-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'=' * 80}\n")

    for fold, fold_results in results.items():
        print(f"{fold}:")
        if hasattr(fold_results, "results_dict"):
            metrics = fold_results.results_dict
            print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")

    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train YOLOv11 baseline with auto-retry")
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specific fold to train (0, 1, or 2). If None, train all folds.",
    )
    parser.add_argument(
        "--model", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="Model size"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cuda, cpu)")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint if available"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retry attempts on error (default: 5)",
    )

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
            resume=args.resume,
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
