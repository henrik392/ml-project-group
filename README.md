# Great Barrier Reef COTS Detection

YOLOv11-based Crown-of-Thorns Starfish detection for the Kaggle TensorFlow Great Barrier Reef competition.

## Setup

```bash
# Install dependencies
uv sync

# Configure Kaggle API (for data download and submissions)
# Place kaggle.json in ~/.kaggle/
```

## Quick Start

Use the Makefile for common tasks:

```bash
# Download and prepare data (all steps)
make data

# Train baseline model
make train

# Train with resume
make resume

# Clean generated files
make clean
```

## Data Preparation

### Step 1: Download Competition Data

```bash
kaggle competitions download -c tensorflow-great-barrier-reef
unzip tensorflow-great-barrier-reef.zip -d data/
```

### Step 2: Convert to YOLO Format

```bash
uv run src/data/prepare_yolo_format.py
```

This creates `data/yolo_format/` with:
- `images/` - All training images
- `labels/` - YOLO format annotations (class x_center y_center width height)

### Step 3: Create Cross-Validation Folds

```bash
uv run src/data/create_folds.py
```

This creates 3-fold splits by video_id in `data/folds/`:
- Fold 0: Train on video_1, video_2 → Validate on video_0
- Fold 1: Train on video_0, video_2 → Validate on video_1
- Fold 2: Train on video_0, video_1 → Validate on video_2

## Training

```bash
# Train single fold
uv run src/training/train_baseline.py --fold 0 --epochs 30 --device mps

# Train all folds
uv run src/training/train_baseline.py --epochs 30 --device mps

# Resume from checkpoint
uv run src/training/train_baseline.py --fold 0 --epochs 30 --device mps --resume

# Train on CPU (more stable but slower)
uv run src/training/train_baseline.py --fold 0 --epochs 30 --device cpu

# Adjust retry attempts (default: 5)
uv run src/training/train_baseline.py --fold 0 --epochs 30 --device mps --max-retries 10
```

### Auto-Retry Feature

Training now includes **automatic retry on errors** with checkpoint recovery:

- **What it does:** If training crashes (tensor mismatches, MPS errors, OOM), it automatically:
  1. Catches the error and logs it
  2. Waits 10 seconds for system recovery
  3. Resumes from the last checkpoint
  4. Retries up to 5 times (configurable with `--max-retries`)

- **Benefits:**
  - No need to manually restart training after crashes
  - Protects against transient MPS backend errors
  - Preserves training progress automatically

- **What it won't retry:**
  - User interrupts (Ctrl+C)
  - Configuration errors (missing files)

## Evaluation

```bash
# Evaluate F2 score
uv run src/evaluation/f2_score.py

# Compare ablation study results
uv run src/evaluation/compare_ablation.py --fold 0
```

## Inference

```bash
# Generate predictions
uv run src/inference/predict.py

# Submit to Kaggle
kaggle competitions submit -c tensorflow-great-barrier-reef -f submission.csv -m "Phase X submission"
```

## Project Structure

```
├── CLAUDE.md          # Development guidelines
├── configs/           # YOLO dataset and training configs
├── src/              # Source code
│   ├── data/         # Data preparation
│   ├── training/     # Training scripts
│   ├── evaluation/   # F2 score evaluation
│   ├── postprocessing/ # Temporal post-processing
│   └── inference/    # Prediction pipeline
├── reports/          # Score charts and documentation
└── analysis/         # Data analysis reports
```

## Approach

Based on the 1st place Kaggle solution with modern improvements:
- **YOLOv11** for object detection
- **SAHI** for small object detection (slice-based inference)
- **3-fold CV** by video_id (leave-one-video-out)
- **Temporal post-processing** (attention area boosting)
- **Underwater augmentations** (MotionBlur, RGBShift, CLAHE)

## Competition

- **Task:** Detect Crown-of-Thorns Starfish (COTS) in underwater video
- **Metric:** F2 Score (emphasizes recall over precision)
- **Data:** 23,501 training frames from 3 videos
