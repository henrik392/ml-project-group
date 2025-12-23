"""
Training utilities for COTS detection.

Reusable components for checkpoint management, retry logic, and training helpers.
"""

import time
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

from ultralytics import YOLO


def with_retry(max_retries: int = 5, wait_time: int = 10):
    """
    Decorator to add automatic retry logic to training functions.

    Automatically resumes from checkpoint on errors and retries up to max_retries times.

    Args:
        max_retries: Maximum number of retry attempts
        wait_time: Seconds to wait between retries

    Example:
        @with_retry(max_retries=5, wait_time=10)
        def train_model(...):
            # training logic
            pass
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0

            while retry_count <= max_retries:
                try:
                    # Execute training function
                    result = func(*args, **kwargs, _retry_count=retry_count)
                    return result

                except KeyboardInterrupt:
                    # User wants to stop - don't retry
                    print("\n\nTraining interrupted by user. Exiting...")
                    raise

                except FileNotFoundError as e:
                    # Configuration errors - don't retry
                    print(f"\n\nConfiguration error: {e}")
                    raise

                except Exception as e:
                    retry_count += 1
                    error_type = type(e).__name__

                    print(f"\n{'!' * 80}")
                    print(f"ERROR [{error_type}]: {str(e)}")
                    print(f"{'!' * 80}\n")

                    if retry_count <= max_retries:
                        print(f"Retry {retry_count}/{max_retries} in {wait_time} seconds...")
                        print("Will automatically resume from checkpoint.\n")
                        time.sleep(wait_time)
                    else:
                        print(f"Max retries ({max_retries}) exceeded. Training failed.")
                        print(f"Last error: [{error_type}] {str(e)}\n")
                        raise

            return None

        return wrapper

    return decorator


def find_latest_checkpoint(
    project: str = "runs/train",
    run_name: str = "yolo11n_fold0",
) -> Optional[Path]:
    """
    Find the most recent checkpoint for a training run.

    Args:
        project: Project directory containing training runs
        run_name: Name pattern to match (e.g., "yolo11n_fold0")

    Returns:
        Path to the latest checkpoint, or None if not found

    Example:
        checkpoint = find_latest_checkpoint("runs/train", "yolo11n_fold0")
        if checkpoint:
            model = YOLO(str(checkpoint))
    """
    potential_checkpoints = list(Path(project).glob(f"{run_name}*/weights/last.pt"))

    if not potential_checkpoints:
        return None

    # Return the most recent checkpoint
    return max(potential_checkpoints, key=lambda p: p.stat().st_mtime)


def load_model_with_checkpoint(
    model_size: str = "n",
    checkpoint_path: Optional[Path] = None,
) -> YOLO:
    """
    Load YOLO model from checkpoint or pretrained weights.

    Args:
        model_size: Model size (n, s, m, l, x)
        checkpoint_path: Path to checkpoint, or None to load pretrained

    Returns:
        Loaded YOLO model

    Example:
        checkpoint = find_latest_checkpoint("runs/train", "yolo11n_fold0")
        model = load_model_with_checkpoint("n", checkpoint)
    """
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading from checkpoint: {checkpoint_path}")
        return YOLO(str(checkpoint_path))
    else:
        print(f"Loading pretrained YOLOv11{model_size} weights")
        return YOLO(f"yolo11{model_size}.pt")


def verify_dataset_config(fold: int) -> str:
    """
    Verify that dataset configuration exists for a fold.

    Args:
        fold: Fold number (0, 1, or 2)

    Returns:
        Path to dataset config

    Raises:
        FileNotFoundError: If config doesn't exist
    """
    data_config = f"configs/dataset_fold_{fold}.yaml"

    if not Path(data_config).exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_config}\n"
            f"Run 'uv run src/data/create_folds.py' first to create fold configs."
        )

    return data_config


def print_training_header(
    fold: int,
    model_size: str,
    phase: str = "Baseline",
    auto_retry: bool = True,
    max_retries: int = 5,
) -> None:
    """
    Print formatted training header with configuration details.

    Args:
        fold: Fold number
        model_size: Model size
        phase: Training phase name
        auto_retry: Whether auto-retry is enabled
        max_retries: Number of retry attempts
    """
    print(f"\n{'=' * 80}")
    print(f"Training YOLOv11{model_size} on Fold {fold} - {phase}")
    if auto_retry:
        print(f"Auto-retry enabled: {max_retries} attempts")
    print(f"{'=' * 80}\n")


def print_training_summary(results, save_dir: Optional[str] = None) -> None:
    """
    Print formatted training summary with final metrics.

    Args:
        results: Training results object
        save_dir: Optional directory where results were saved
    """
    print(f"\n{'=' * 80}")
    print("Training complete!")
    if save_dir:
        print(f"Results saved to: {save_dir}")
    print(f"{'=' * 80}\n")

    # Print final metrics
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        print("Final Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


def get_run_name(model_size: str, fold: int, phase: str = "") -> str:
    """
    Generate standardized run name for training.

    Args:
        model_size: Model size (n, s, m, l, x)
        fold: Fold number
        phase: Optional phase suffix (e.g., "phase2", "optimized")

    Returns:
        Run name string

    Example:
        get_run_name("n", 0) -> "yolo11n_fold0"
        get_run_name("n", 0, "phase2") -> "yolo11n_fold0_phase2"
    """
    base_name = f"yolo11{model_size}_fold{fold}"
    if phase:
        return f"{base_name}_{phase}"
    return base_name
