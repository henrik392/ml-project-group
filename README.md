# ML Project Group - TensorFlow Great Barrier Reef

Object detection project using the TensorFlow Great Barrier Reef competition dataset.

## Setup

### Prerequisites

- Python >= 3.14
- Kaggle API credentials (for dataset download)

### Installation

1. Install dependencies:
```bash
uv sync
```

2. Download the dataset:

**Option A: Using Kaggle CLI (recommended)**
```bash
kaggle competitions download -c tensorflow-great-barrier-reef
unzip tensorflow-great-barrier-reef.zip -d data/
```

**Option B: Manual download**
- Download the dataset from [Kaggle](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef)
- Place the zip file in the project root
- Extract to `data/` folder:
```bash
unzip tensorflow-great-barrier-reef.zip -d data/
```

### Dataset Structure

After extraction, the `data/` directory contains:
- `train_images/` - Training video frames organized by video ID
- `train.csv` - Training annotations
- `test.csv` - Test set metadata
- `example_sample_submission.csv` - Sample submission format
- `greatbarrierreef/` - Competition utilities

## Development

Run the main script:
```bash
uv run python main.py
```
