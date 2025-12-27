# Experiment Results

This directory contains experiment results in a standardized CSV format.

## Results CSV Schema

**File**: `results.csv`

### Column Definitions

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `experiment_id` | string | Unique experiment identifier | `exp01_yolov5` |
| `fold_id` | int | Cross-validation fold (0, 1, 2) | `0` |
| `eval_video_id` | string | Video used for evaluation | `video_0` |
| `model` | string | Model architecture | `yolov11n` |
| `inference` | string | Inference mode | `standard` or `sahi` |
| `tracking` | string | Tracking method | `none` or `bytetrack` |
| `conf` | float | Confidence threshold | `0.25` |
| `iou` | float | NMS IoU threshold | `0.45` |
| `imgsz` | int | Image size (pixels) | `640` |
| `f2` | float | F2 score (primary metric) | `0.65` |
| `map50` | float | mAP@IoU=0.50 | `0.35` |
| `recall` | float | Recall | `0.55` |
| `precision` | float | Precision | `0.42` |
| `ms_per_frame` | float | Latency (milliseconds/frame) | `22.5` |
| `seed` | int | Random seed for reproducibility | `42` |
| `timestamp` | string | ISO 8601 timestamp | `2025-12-27T10:00:00` |

### Example Entry

```csv
exp02_yolov11,0,video_0,yolov11n,standard,none,0.25,0.45,640,0.42,0.35,0.55,0.38,22.2,42,2025-12-27T10:00:00
```

## Usage

### Append Results

```python
import csv
from datetime import datetime

result = {
    'experiment_id': 'exp02_yolov11',
    'fold_id': 0,
    'eval_video_id': 'video_0',
    'model': 'yolov11n',
    'inference': 'standard',
    'tracking': 'none',
    'conf': 0.25,
    'iou': 0.45,
    'imgsz': 640,
    'f2': 0.42,
    'map50': 0.35,
    'recall': 0.55,
    'precision': 0.38,
    'ms_per_frame': 22.2,
    'seed': 42,
    'timestamp': datetime.now().isoformat()
}

with open('outputs/metrics/results.csv', 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=result.keys())
    writer.writerow(result)
```

### Load Results

```python
import pandas as pd

df = pd.read_csv('outputs/metrics/results.csv')

# Filter by experiment
exp02 = df[df['experiment_id'] == 'exp02_yolov11']

# Compare experiments
comparison = df.groupby('experiment_id')['f2'].mean().sort_values(ascending=False)
```

## Notes

- **Primary metric**: F2 score (recall-weighted, 5× emphasis on recall)
- **Latency**: `ms_per_frame` is more stable than FPS across different hardware
- **Fold strategy**: Leave-one-video-out cross-validation (3 folds)
- **Reproducibility**: Always set `seed` for deterministic results
