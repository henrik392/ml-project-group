# Great Barrier Reef COTS Detection

## Project Context
- Competition: TensorFlow Great Barrier Reef
- Task: Detect Crown-of-Thorns Starfish (COTS) in underwater video sequences
- Metric: F2 Score (recall-focused)
- Evaluation: 3-fold cross-validation by video_id

## Development Guidelines
- Use YOLOv11 via ultralytics library
- Train with `device='mps'` on M4 Max
- Always validate with 3-fold video-based CV (leave-one-video-out)
- Use the winning team's F2 algorithm for consistency
- Create branch and PR for each phase
- Document all scores and experiments
- No "contributed by claude" in commits

## Key Files
- `configs/dataset.yaml` - YOLO dataset config
- `src/data/prepare_yolo_format.py` - Data conversion
- `src/training/train.py` - Training script
- `src/evaluation/f2_score.py` - F2 evaluation
- `src/postprocessing/temporal_boost.py` - Temporal post-processing
- `reports/` - Score charts and documentation

## Commands
```bash
# Train
python src/training/train.py

# Evaluate
python src/evaluation/f2_score.py

# Predict
python src/inference/predict.py

# Submit to Kaggle
kaggle competitions submit -c tensorflow-great-barrier-reef -f submission.csv -m "Phase X submission"
```

## Winning Solution Key Techniques
- 3-fold CV by video_id (Trust CV)
- Modified hyperparams: box=0.2, iou_t=0.3
- Augmentations: rotation, mixup, Transpose (NO HSV)
- SAHI for small object detection
- Temporal post-processing (attention area boosting)
- Classification re-scoring (optional, Phase 5)

## Validation Strategy
```
Fold 0: Train on video_1, video_2 → Validate on video_0
Fold 1: Train on video_0, video_2 → Validate on video_1
Fold 2: Train on video_0, video_1 → Validate on video_2
```

## Target Scores
- Phase 1: Any valid CV F2 (baseline)
- Phase 2: CV F2 > 0.65 (optimized detection)
- Phase 3: CV F2 > 0.68 (temporal post-processing)
- Phase 4: CV F2 > 0.70 (final model)
- Phase 5: CV F2 > 0.72 (optional classification re-scoring)
