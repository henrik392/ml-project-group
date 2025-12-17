# Great Barrier Reef COTS Detection

YOLOv11-based Crown-of-Thorns Starfish detection for the Kaggle TensorFlow Great Barrier Reef competition.

## Setup

```bash
# Install dependencies
uv sync

# Configure Kaggle API (for data download and submissions)
# Place kaggle.json in ~/.kaggle/
```

## Usage

```bash
# Train
python src/training/train.py

# Evaluate
python src/evaluation/f2_score.py

# Predict
python src/inference/predict.py

# Submit to Kaggle
kaggle competitions submit -c tensorflow-great-barrier-reef -f submission.csv -m "Description"
```

## Data

Download from Kaggle and place in `data/` directory:
```bash
kaggle competitions download -c tensorflow-great-barrier-reef
unzip tensorflow-great-barrier-reef.zip -d data/
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
