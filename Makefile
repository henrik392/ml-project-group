.PHONY: help data download prepare-yolo create-folds exp-baseline exp-yolov5 exp-sahi exp-bytetrack overnight video video-test clean clean-runs clean-data

# Default target
help:
	@echo "Great Barrier Reef COTS Detection - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make data            - Download and prepare all data (download + prepare-yolo + create-folds)"
	@echo "  make download        - Download competition data from Kaggle"
	@echo "  make prepare-yolo    - Convert data to YOLO format"
	@echo "  make create-folds    - Create 3-fold CV splits by video_id"
	@echo ""
	@echo "  make exp-baseline    - Run YOLOv11 baseline experiment (exp02)"
	@echo "  make exp-yolov5      - Run YOLOv5 baseline experiment (exp01)"
	@echo "  make exp-sahi        - Run SAHI inference experiment (exp06)"
	@echo "  make exp-bytetrack   - Run ByteTrack experiment (exp07)"
	@echo "  make overnight       - Run multiple experiments sequentially (edit run_overnight.sh)"
	@echo ""
	@echo "  make video           - Generate 2x2 comparison video (outputs/videos/comparison_4up.mp4)"
	@echo "  make video-test      - Generate test video with 100 frames (faster)"
	@echo ""
	@echo "  make clean           - Clean all generated files"
	@echo "  make clean-runs      - Clean training runs only"
	@echo "  make clean-data      - Clean processed data only"

# Data pipeline
data: download prepare-yolo create-folds
	@echo "✓ Data preparation complete!"

download:
	@echo "Downloading competition data..."
	@mkdir -p data
	@if [ ! -f data/train.csv ]; then \
		kaggle competitions download -c tensorflow-great-barrier-reef && \
		unzip -q tensorflow-great-barrier-reef.zip -d data/ && \
		rm tensorflow-great-barrier-reef.zip; \
	else \
		echo "Data already downloaded (train.csv exists)"; \
	fi

prepare-yolo:
	@echo "Converting data to YOLO format..."
	@uv run src/data/prepare_yolo_format.py

create-folds:
	@echo "Creating 3-fold CV splits..."
	@uv run src/data/create_folds.py

# Experiments
exp-baseline:
	@echo "Running YOLOv11 baseline experiment..."
	@uv run python -m src.run_experiment --config configs/experiments/exp02_yolov11_baseline.yaml

exp-yolov5:
	@echo "Running YOLOv5 baseline experiment..."
	@uv run python -m src.run_experiment --config configs/experiments/exp01_yolov5_baseline.yaml

exp-sahi:
	@echo "Running SAHI inference experiment..."
	@uv run python -m src.run_experiment --config configs/experiments/exp06_sahi_inference.yaml

exp-bytetrack:
	@echo "Running ByteTrack experiment..."
	@uv run python -m src.run_experiment --config configs/experiments/exp07_bytetrack.yaml

# Overnight batch processing
overnight:
	@echo "Running overnight experiments (see run_overnight.sh for configuration)..."
	@./run_overnight.sh

# Evaluation
evaluate:
	@echo "Evaluating F2 score..."
	@uv run src/evaluation/f2_score.py

# Video generation
video:
	@echo "Generating 2x2 comparison video (this may take 30-60 minutes)..."
	@uv run python -m src.visualization.video_grid
	@echo "✓ Video saved to outputs/videos/comparison_4up.mp4"

video-test:
	@echo "Generating test video with 100 frames..."
	@uv run python -m src.visualization.video_grid --max-frames 100
	@echo "✓ Test video saved to outputs/videos/comparison_4up.mp4"

# Inference
predict:
	@echo "Generating predictions..."
	@uv run src/inference/predict.py

# Cleanup
clean: clean-runs clean-data
	@echo "✓ All generated files cleaned!"

clean-runs:
	@echo "Cleaning training runs..."
	@rm -rf runs/

clean-data:
	@echo "Cleaning processed data..."
	@rm -rf data/yolo_format/ data/folds/
