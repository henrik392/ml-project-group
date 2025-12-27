"""
Baseline YOLOv11 training script for COTS detection.

Supports config-based training with automatic retry on errors.
"""

from pathlib import Path

from ultralytics import YOLO

from src.training.utils import (
    find_latest_checkpoint,
    with_retry,
)


@with_retry(max_retries=5, wait_time=10)
def train_from_config(
    config: dict,
    _retry_count: int = 0,
) -> dict:
    """
    Train YOLOv11 from experiment config with automatic retry on errors.

    Args:
        config: Experiment configuration dict with keys:
            - model: Model name (yolov11n, yolov5n, etc.)
            - data: Path to dataset YAML
            - fold_id: Fold number (0, 1, or 2)
            - epochs: Number of training epochs
            - imgsz: Image size for training
            - batch: Batch size
            - device: Device to use (default: 'mps')
            - hyperparameters: Optional dict with box, cls, dfl, etc.
            - augmentation: Optional dict with hsv_h, hsv_s, degrees, etc.
        _retry_count: Internal retry counter (set by decorator)

    Returns:
        Dict with training results including weights path and metrics
    """
    # Extract config values with defaults
    model_name = config.get("model", "yolov11n")
    data_config = config.get(
        "data", f"configs/dataset_fold_{config.get('fold_id', 0)}.yaml"
    )
    fold_id = config.get("fold_id", 0)
    epochs = config.get("epochs", 50)
    imgsz = config.get("imgsz", 640)
    batch = config.get("batch", 16)
    device = config.get("device", "mps")
    project = config.get("project", "runs/train")
    resume = config.get("resume", False)

    # Print training header
    print(f"\n{'=' * 80}")
    print(f"Training {model_name} - Fold {fold_id}")
    print(f"{'=' * 80}")
    print(f"Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}, Device: {device}")
    print("Auto-retry: Enabled (max 5 retries)")
    print(f"{'=' * 80}\n")

    # Verify dataset config exists
    if not Path(data_config).exists():
        raise FileNotFoundError(f"Dataset config not found: {data_config}")

    # Find checkpoint if resuming or retrying
    run_name = f"{model_name}_fold{fold_id}"
    checkpoint_path = None
    if resume or _retry_count > 0:
        checkpoint_path = find_latest_checkpoint(project, run_name)
        if checkpoint_path:
            if _retry_count > 0:
                print(
                    f"\n[RETRY {_retry_count}/5] Resuming from checkpoint: {checkpoint_path}"
                )
            else:
                print(f"Resuming from checkpoint: {checkpoint_path}")
        elif resume:
            print("No checkpoint found, starting from pretrained weights")

    # Load model
    if checkpoint_path:
        print(f"Loading from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        print(f"Loading pretrained {model_name}")
        model = YOLO(f"{model_name}.pt")

    # Build training kwargs
    train_kwargs = {
        "data": data_config,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": project,
        "name": run_name,
        "exist_ok": True,
        "verbose": True,
    }

    # Add hyperparameters if provided
    if "hyperparameters" in config:
        train_kwargs.update(config["hyperparameters"])

    # Add augmentation settings if provided
    if "augmentation" in config:
        train_kwargs.update(config["augmentation"])

    # Set resume flag if we have a checkpoint
    if checkpoint_path:
        train_kwargs["resume"] = True

    # Train
    print("\nStarting training...")
    results = model.train(**train_kwargs)

    # Get weights path
    weights_path = str(Path(results.save_dir) / "weights" / "best.pt")

    # Extract metrics from results
    metrics = {
        "weights": weights_path,
        "save_dir": str(results.save_dir),
    }

    if hasattr(results, "results_dict"):
        metrics.update(
            {
                "map50": results.results_dict.get("metrics/mAP50(B)", 0),
                "map50_95": results.results_dict.get("metrics/mAP50-95(B)", 0),
            }
        )

    print(f"\n{'=' * 80}")
    print("Training Complete")
    print(f"Weights: {weights_path}")
    print(f"Save dir: {results.save_dir}")
    print(f"{'=' * 80}\n")

    return metrics
