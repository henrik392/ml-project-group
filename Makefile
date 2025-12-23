.PHONY: help data download prepare-yolo create-folds train resume train-all clean clean-runs clean-data

# Default target
help:
	@echo "Great Barrier Reef COTS Detection - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make data          - Download and prepare all data (download + prepare-yolo + create-folds)"
	@echo "  make download      - Download competition data from Kaggle"
	@echo "  make prepare-yolo  - Convert data to YOLO format"
	@echo "  make create-folds  - Create 3-fold CV splits by video_id"
	@echo ""
	@echo "  make train         - Train fold 0 baseline (30 epochs, MPS)"
	@echo "  make resume        - Resume training fold 0 from checkpoint"
	@echo "  make train-all     - Train all 3 folds"
	@echo ""
	@echo "  make clean         - Clean all generated files"
	@echo "  make clean-runs    - Clean training runs only"
	@echo "  make clean-data    - Clean processed data only"

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

# Training
train:
	@echo "Training fold 0 baseline (30 epochs, MPS)..."
	@uv run src/training/train_baseline.py --fold 0 --epochs 30 --device mps

resume:
	@echo "Resuming training fold 0 from checkpoint..."
	@uv run src/training/train_baseline.py --fold 0 --epochs 30 --device mps --resume

train-all:
	@echo "Training all 3 folds (30 epochs each, MPS)..."
	@uv run src/training/train_baseline.py --epochs 30 --device mps

# Evaluation
evaluate:
	@echo "Evaluating F2 score..."
	@uv run src/evaluation/f2_score.py

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
