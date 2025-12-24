"""
Training configuration classes for COTS detection.

Provides reusable configuration classes for different training phases.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Base configuration for YOLO training.

    This can be extended for different training phases (baseline, optimized, etc.).
    """

    # Model settings
    model_size: str = "n"  # n, s, m, l, x
    imgsz: int = 640  # Image size for training
    batch: int = 16  # Batch size

    # Training settings
    epochs: int = 50  # Number of training epochs
    patience: int = 10  # Early stopping patience
    device: str = "mps"  # Device: mps, cuda, cpu

    # Data settings
    fold: int = 0  # Fold number (0, 1, 2)

    # Output settings
    project: str = "runs/train"  # Project directory
    phase: str = ""  # Phase suffix for run name (e.g., "phase2")

    # Advanced settings
    amp: bool = True  # Automatic Mixed Precision
    rect: bool = False  # Rectangular training (disabled to prevent tensor mismatches)
    save: bool = True  # Save checkpoints
    plots: bool = True  # Generate training plots
    val: bool = True  # Run validation
    verbose: bool = True  # Verbose output

    # Resume settings
    resume: bool = False  # Resume from checkpoint
    max_retries: int = 5  # Maximum retry attempts on error

    # Hyperparameters (can be overridden in subclasses)
    hyperparameters: dict = field(default_factory=dict)

    def get_run_name(self) -> str:
        """Generate run name for this configuration."""
        base_name = f"yolo11{self.model_size}_fold{self.fold}"
        if self.phase:
            return f"{base_name}_{self.phase}"
        return base_name

    def get_data_config(self) -> str:
        """Get path to dataset configuration file."""
        return f"configs/dataset_fold_{self.fold}.yaml"

    def to_train_kwargs(self) -> dict:
        """
        Convert config to kwargs for model.train().

        Returns:
            Dictionary of training arguments for YOLO.train()
        """
        kwargs = {
            "data": self.get_data_config(),
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device,
            "project": self.project,
            "name": self.get_run_name(),
            "patience": self.patience,
            "save": self.save,
            "plots": self.plots,
            "val": self.val,
            "amp": self.amp,
            "verbose": self.verbose,
            "rect": self.rect,
        }

        # Add any custom hyperparameters
        kwargs.update(self.hyperparameters)

        return kwargs


@dataclass
class BaselineConfig(TrainingConfig):
    """
    Phase 1: Baseline configuration with default hyperparameters.

    Simple baseline to establish performance with standard YOLO settings.
    """

    phase: str = "baseline"


@dataclass
class OptimizedConfig(TrainingConfig):
    """
    Phase 2: Optimized configuration based on winning solution.

    Incorporates competition-winning hyperparameters:
    - box=0.2 (box loss weight)
    - iou_t=0.3 (IoU training threshold)
    - Optimized augmentations
    """

    phase: str = "optimized"
    hyperparameters: dict = field(
        default_factory=lambda: {
            "box": 0.2,  # Box loss weight (winning solution)
            "cls": 0.5,  # Classification loss weight
            "dfl": 1.5,  # DFL loss weight
            "iou": 0.3,  # IoU training threshold (winning solution)
            # Augmentations
            "degrees": 15.0,  # Rotation degrees
            "translate": 0.1,  # Translation
            "scale": 0.5,  # Scale
            "shear": 0.0,  # Shear
            "perspective": 0.0,  # Perspective
            "flipud": 0.5,  # Vertical flip probability
            "fliplr": 0.5,  # Horizontal flip probability
            "mosaic": 1.0,  # Mosaic augmentation
            "mixup": 0.15,  # Mixup augmentation (winning solution)
            "copy_paste": 0.0,  # Copy-paste augmentation
            # Note: HSV augmentation disabled per winning solution
            "hsv_h": 0.0,  # Hue augmentation
            "hsv_s": 0.0,  # Saturation augmentation
            "hsv_v": 0.0,  # Value augmentation
        }
    )


@dataclass
class TemporalConfig(OptimizedConfig):
    """
    Phase 3: Temporal configuration with post-processing.

    Extends optimized config for temporal post-processing experiments.
    """

    phase: str = "temporal"


@dataclass
class FinalConfig(OptimizedConfig):
    """
    Phase 4: Final configuration with all optimizations.

    Best settings from all previous phases.
    """

    phase: str = "final"
    epochs: int = 100  # Longer training for final model


def get_config(phase: str = "baseline", **kwargs) -> TrainingConfig:
    """
    Factory function to get training configuration by phase.

    Args:
        phase: Training phase ("baseline", "optimized", "temporal", "final")
        **kwargs: Override any config parameters

    Returns:
        TrainingConfig instance

    Example:
        config = get_config("optimized", fold=1, epochs=30)
        kwargs = config.to_train_kwargs()
    """
    configs = {
        "baseline": BaselineConfig,
        "optimized": OptimizedConfig,
        "temporal": TemporalConfig,
        "final": FinalConfig,
    }

    config_class = configs.get(phase.lower(), BaselineConfig)
    return config_class(**kwargs)
