# Great Barrier Reef COTS Detection

**Tsinghua University - Machine Learning Course Project**

Modern object detection approaches for Crown-of-Thorns Starfish (COTS) in underwater video: YOLOv5 → YOLOv11 → SAHI → ByteTrack.

## Academic Context

- **Course**: Machine Learning, Tsinghua University
- **Project Type**: Computer vision object detection research
- **Dataset**: Kaggle TensorFlow Great Barrier Reef (23,501 underwater video frames, 3 videos)
- **Task**: Detect COTS with emphasis on recall (F2 metric)
- **Evaluation**: Leave-one-video-out cross-validation (3 folds, no frame leakage)
- **Primary Metric**: F2 Score (recall-weighted, 5× emphasis on recall over precision)

## Course Deliverables

| Deliverable | Deadline | Status |
|-------------|----------|--------|
| Poster/Demo (voting) | **12/28 (Sun) 24:00** | Pending |
| Presentation + Poster | **12/30 (Tue) 24:00** | Pending |
| **Final Report + Code** | **01/07 (Wed) 24:00** | In Progress |

**Report**: NeurIPS LaTeX format, English, sections: Problem → Motivation → Methods → Results → Conclusions

## Setup

```bash
# Install dependencies
uv sync

# Configure Kaggle API (for data download and submissions)
# Place kaggle.json in ~/.kaggle/
```

## Experimental Storyline

This project follows a clear narrative arc for the final report:

1. **Historical Baseline**: YOLOv5 (2022 SOTA) vs. YOLOv11 (modern) - same split, same metric
2. **YOLOv11 Ablation**: Minimal sweeps (conf, iou, imgsz) to establish strong baseline
3. **Inference Strategy**: Standard vs. SAHI tiled inference (accuracy/speed tradeoff)
4. **Temporal Context**: ByteTrack (`model.track`) for video recall enhancement
5. **Qualitative Analysis**: Visual comparisons and failure case analysis

**Key Principles**:
- **Report = single source of truth** (slides/poster derived from it)
- **Clarity > complexity** (if defaults win, we report that and move on)
- **F2 everywhere** (all results in F2, thresholds optimized for F2)
- **Freeze training** (train once per fold → vary only inference settings)

## Quick Start

```bash
# 1. Download and prepare data
make data

# 2. Run an experiment
uv run src/run_experiment.py --config configs/experiments/exp02_yolov11_baseline.yaml
# Or use shortcuts: make exp-baseline

# 3. View all results
cat outputs/metrics/results.csv
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

### Step 3: Create Leave-One-Video-Out Folds

```bash
uv run src/data/create_folds.py
```

This creates **leave-one-video-out cross-validation (3 folds)** in `data/folds/`:
- Fold 0: Train on video_1, video_2 → Eval on video_0
- Fold 1: Train on video_0, video_2 → Eval on video_1
- Fold 2: Train on video_0, video_1 → Eval on video_2

**Rationale**: Video frames are temporally correlated. Holding out full videos prevents data leakage and ensures realistic evaluation.

## Running Experiments

All experiments use standardized YAML configs in `configs/experiments/`. The runner handles training, inference, evaluation, and result logging.

```bash
# Run a single experiment
uv run src/run_experiment.py --config configs/experiments/exp02_yolov11_baseline.yaml

# Preview experiment without executing
uv run src/run_experiment.py --config configs/experiments/exp01_yolov5_baseline.yaml --dry-run

# Quick shortcuts via Makefile
make exp-baseline    # YOLOv11 baseline
make exp-yolov5      # YOLOv5 baseline
make exp-sahi        # SAHI inference
make exp-bytetrack   # ByteTrack tracking
```

### Experiment Workflow

Each experiment automatically runs:
1. **Training** (or loads existing weights if `skip_training: true`)
2. **Inference** (standard, SAHI, or with ByteTrack)
3. **Evaluation** (F2, mAP50, recall, precision, latency)
4. **Results logging** to `outputs/metrics/results.csv`

### Available Experiments

- **exp01_yolov5_baseline.yaml** - Historical baseline (2022 SOTA)
- **exp02_yolov11_baseline.yaml** - Modern YOLOv11 baseline
- **exp03_conf_sweep.yaml** - Confidence threshold sweep
- **exp04_iou_sweep.yaml** - IoU threshold sweep
- **exp05_resolution_scaling.yaml** - Resolution comparison (640 vs 1280)
- **exp06_sahi_inference.yaml** - SAHI tiled inference
- **exp07_bytetrack.yaml** - ByteTrack temporal tracking
- **exp08_sahi_bytetrack.yaml** - Combined SAHI + ByteTrack

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
├── CLAUDE.md                    # Development guidelines
├── README.md                    # This file
├── configs/
│   ├── dataset.yaml             # YOLO dataset config
│   └── experiments/             # Experiment configs (exp01.yaml, exp02.yaml, ...)
├── src/
│   ├── run_experiment.py        # Main runner entrypoint
│   ├── data/                    # Data preparation scripts
│   ├── training/                # Training scripts
│   ├── inference/               # Inference (standard, SAHI, ByteTrack)
│   ├── evaluation/              # F2 score evaluation
│   └── visualization/           # Report figure generation
├── outputs/
│   ├── runs/                    # Training outputs (model weights, curves)
│   ├── metrics/                 # results.csv (all experiment results)
│   ├── figures/                 # Report-ready figures
│   └── videos/                  # Comparison videos (4-up grids)
└── reports/
    ├── final_report.md          # **SINGLE SOURCE OF TRUTH**
    ├── figures/                 # Training curves, ablation plots
    ├── ablation_study_analysis.md   # Prior ablation study (appendix reference)
    └── *.md                     # Other analysis documents
```

## Experiment Matrix

| ID | Model | Inference | Tracking | Purpose |
|----|-------|-----------|----------|---------|
| E1 | YOLOv5n | Standard | None | Historical baseline (2022 SOTA) |
| E2 | YOLOv11n | Standard | None | Modern baseline |
| E3 | YOLOv11n | Standard | None | Conf threshold sweep for F2 |
| E4 | YOLOv11n | Standard | None | IoU threshold sweep |
| E5 | YOLOv11n | Standard | None | Resolution sweep (640 vs 1280) |
| E6 | YOLOv11n | SAHI | None | Tiled inference (accuracy/speed) |
| E7 | YOLOv11n | Standard | ByteTrack | Temporal context |
| E8 | YOLOv11n | SAHI | ByteTrack | Combined best methods |

**Results**: See `outputs/metrics/results.csv` or **`reports/final_report.md`** (single source of truth)

## Results CSV Schema

```csv
experiment_id,fold_id,eval_video_id,model,inference,tracking,conf,iou,imgsz,f2,map50,recall,precision,ms_per_frame,seed,timestamp
exp01_yolov5,0,video_0,yolov5n,standard,none,0.25,0.45,640,0.42,0.35,0.55,0.38,22.2,42,2025-12-27T10:00:00
```

**Column definitions**:
- `fold_id`: Which fold (0, 1, 2)
- `eval_video_id`: Which video held out (video_0, video_1, video_2)
- `ms_per_frame`: Latency in milliseconds (more stable than FPS)
- `seed`: Random seed for reproducibility
