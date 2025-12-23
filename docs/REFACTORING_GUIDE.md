# Training Code Refactoring Guide

## Overview

The training codebase has been refactored into reusable, modular components to support multiple training phases without code duplication.

## Architecture

### 1. **Core Modules**

#### `src/training/utils.py`
Reusable utilities for all training scripts:

- **`@with_retry(max_retries, wait_time)`** - Decorator for automatic retry on errors
- **`find_latest_checkpoint(project, run_name)`** - Find most recent checkpoint
- **`load_model_with_checkpoint(model_size, checkpoint_path)`** - Load model from checkpoint or pretrained
- **`verify_dataset_config(fold)`** - Verify dataset configuration exists
- **`print_training_header(...)`** - Formatted training header
- **`print_training_summary(results, save_dir)`** - Formatted training summary
- **`get_run_name(model_size, fold, phase)`** - Generate standardized run names

#### `src/training/config.py`
Configuration classes for different training phases:

- **`TrainingConfig`** - Base configuration dataclass
- **`BaselineConfig`** - Phase 1: Default hyperparameters
- **`OptimizedConfig`** - Phase 2: Winning solution hyperparameters
- **`TemporalConfig`** - Phase 3: Temporal post-processing
- **`FinalConfig`** - Phase 4: Final optimized model

**Key features:**
- `to_train_kwargs()` - Convert config to YOLO training arguments
- `get_run_name()` - Generate run name for the phase
- `get_data_config()` - Get dataset configuration path
- Easy to extend with new hyperparameters

### 2. **Training Scripts**

#### `src/training/train_baseline.py`
Phase 1 baseline training with default settings.

#### `src/training/train_optimized.py`
Phase 2 optimized training with winning solution hyperparameters.

Both scripts share the same clean structure:
```python
from src.training.config import BaselineConfig  # or OptimizedConfig
from src.training.utils import (
    find_latest_checkpoint,
    load_model_with_checkpoint,
    print_training_header,
    print_training_summary,
    verify_dataset_config,
    with_retry,
)

@with_retry(max_retries=5, wait_time=10)
def train_fold(..., _retry_count=0):
    config = BaselineConfig(...)  # or OptimizedConfig
    # ... checkpoint handling
    model = load_model_with_checkpoint(model_size, checkpoint_path)
    train_kwargs = config.to_train_kwargs()
    results = model.train(**train_kwargs)
    return results
```

## Benefits of Refactoring

### ✅ **Code Reusability**
- Retry logic implemented once, reused everywhere
- Checkpoint management unified
- No code duplication across phases

### ✅ **Easy to Extend**
Creating a new training phase is now trivial:

1. **Add configuration** to `config.py`:
```python
@dataclass
class Phase3Config(OptimizedConfig):
    phase: str = "phase3"
    hyperparameters: dict = field(
        default_factory=lambda: {
            # your custom hyperparameters
        }
    )
```

2. **Create training script** `train_phase3.py`:
```python
from src.training.config import Phase3Config
from src.training.utils import *

@with_retry(max_retries=5)
def train_fold(...):
    config = Phase3Config(...)
    # ... same structure as baseline
```

That's it! ~20 lines vs ~200 lines of duplicated code.

### ✅ **Maintainability**
- Bug fixes in retry logic → fixed everywhere
- Improvements to checkpoint handling → benefit all scripts
- Consistent behavior across all phases

### ✅ **Type Safety & Documentation**
- Dataclass configs provide type hints
- Clear documentation in docstrings
- IDE autocomplete support

## Usage Examples

### Using Configuration Classes

```python
# Create baseline config
config = BaselineConfig(
    fold=0,
    model_size="n",
    epochs=30,
    device="mps"
)

# Get training kwargs
kwargs = config.to_train_kwargs()
# Returns: {data: ..., epochs: 30, device: "mps", ...}

# Get run name
config.get_run_name()  # "yolo11n_fold0_baseline"
```

### Using Optimized Config

```python
config = OptimizedConfig(fold=1, epochs=50)

# Hyperparameters from winning solution are pre-configured
config.hyperparameters
# Returns: {box: 0.2, iou: 0.3, mixup: 0.15, ...}
```

### Using Utilities

```python
from src.training.utils import find_latest_checkpoint

# Find checkpoint for a run
checkpoint = find_latest_checkpoint("runs/train", "yolo11n_fold0_baseline")
if checkpoint:
    model = YOLO(str(checkpoint))
```

### Using Retry Decorator

```python
@with_retry(max_retries=3, wait_time=5)
def my_training_function(..., _retry_count=0):
    # _retry_count is automatically passed by decorator
    if _retry_count > 0:
        print(f"Retry {_retry_count}")
    # ... training code
```

## Migration Guide

### Old Style (Duplicated Code)
```python
# train_phase1.py - 200 lines
def train_fold(...):
    # 50 lines of retry logic
    # 30 lines of checkpoint handling
    # 20 lines of config verification
    # 50 lines of training
    # 20 lines of result printing

# train_phase2.py - 200 lines (duplicated!)
def train_fold(...):
    # same 50 lines of retry logic
    # same 30 lines of checkpoint handling
    # ...
```

### New Style (Modular)
```python
# train_phase1.py - 50 lines
from src.training.config import BaselineConfig
from src.training.utils import *

@with_retry(max_retries=5)
def train_fold(..., _retry_count=0):
    config = BaselineConfig(...)
    # ... 20 lines of core logic
```

## Adding New Training Phases

### Step 1: Define Configuration

```python
# In src/training/config.py
@dataclass
class MyNewPhaseConfig(TrainingConfig):
    phase: str = "my_phase"

    hyperparameters: dict = field(
        default_factory=lambda: {
            "box": 0.3,
            "my_custom_param": 42,
            # ...
        }
    )
```

### Step 2: Create Training Script

```python
# src/training/train_my_phase.py
import argparse
from src.training.config import MyNewPhaseConfig
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
    fold=0, model_size="n", epochs=50, ..., _retry_count=0
):
    config = MyNewPhaseConfig(fold=fold, model_size=model_size, ...)
    verify_dataset_config(fold)

    checkpoint = find_latest_checkpoint("runs/train", config.get_run_name())
    model = load_model_with_checkpoint(model_size, checkpoint)

    results = model.train(**config.to_train_kwargs())
    print_training_summary(results, results.save_dir)
    return results

def main():
    parser = argparse.ArgumentParser(description="My new phase")
    # ... standard arguments
    args = parser.parse_args()
    train_fold(fold=args.fold, ...)

if __name__ == "__main__":
    main()
```

### Step 3: Update Makefile (Optional)

```makefile
train-my-phase:
    @echo "Training my new phase..."
    @uv run src/training/train_my_phase.py --fold 0 --epochs 50
```

## Testing

All utilities have clear interfaces and can be tested independently:

```python
# Test checkpoint finding
checkpoint = find_latest_checkpoint("runs/train", "yolo11n_fold0")
assert checkpoint.exists()

# Test config generation
config = BaselineConfig(fold=0)
kwargs = config.to_train_kwargs()
assert kwargs["data"] == "configs/dataset_fold_0.yaml"
```

## Summary

This refactoring transforms the training codebase from:
- ❌ **Monolithic, duplicated scripts** (200+ lines each)
- ❌ **Manual retry logic everywhere**
- ❌ **Hardcoded configurations**

To:
- ✅ **Modular, reusable components** (50 lines per script)
- ✅ **Automatic retry with decorator**
- ✅ **Declarative configurations**
- ✅ **Easy to extend for new phases**

Creating a new training phase now takes **5 minutes** instead of **1 hour** of copy-pasting and modifying code.
