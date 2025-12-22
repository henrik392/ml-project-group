# Great Barrier Reef COTS Detection

## Project Context
- Competition: TensorFlow Great Barrier Reef
- Task: Detect Crown-of-Thorns Starfish (COTS) in underwater video sequences
- Metric: F2 Score (recall-focused)
- Evaluation: 3-fold cross-validation by video_id

## Development Guidelines
- **ALWAYS use `uv run` to execute Python scripts** (not `python3` or `python` directly)
- Use YOLOv11 via ultralytics library
- Train with `device='mps'` on M4 Max (or cloud GPUs if MPS has compatibility issues)
- Always validate with 3-fold video-based CV (leave-one-video-out)
- Use the winning team's F2 algorithm for consistency
- Create branch and PR for each phase
- Document all scores and experiments
- No "contributed by claude" in commits

## Final Report & Presentation Requirements

**IMPORTANT**: Continuously generate content for the final report and presentation throughout development.

### Report Content to Generate

After each phase, create/update:

1. **Methodology Section**
   - Architecture diagrams (model, pipeline)
   - Hyperparameter tables
   - Augmentation strategies
   - Cross-validation approach

2. **Results Section**
   - Performance comparison tables (Phase 1 vs Phase 2 vs...)
   - Training curves (loss, mAP, precision, recall)
   - Confusion matrices
   - PR curves, F1 curves
   - Example predictions (success & failure cases)

3. **Analysis Section**
   - Ablation studies
   - Error analysis
   - Lessons learned
   - Technical challenges & solutions

4. **Figures & Visualizations**
   - High-quality plots (use matplotlib/seaborn)
   - Example detections with bounding boxes
   - Before/after comparisons
   - Architecture diagrams

### Presentation Slides to Prepare

1. **Problem Overview** (1-2 slides)
   - COTS detection importance
   - Dataset characteristics
   - Evaluation metric (F2)

2. **Methodology** (2-3 slides)
   - YOLOv11 architecture
   - Training strategy
   - Cross-validation approach
   - Key techniques from winning solution

3. **Results** (3-4 slides)
   - Performance progression (baseline → optimized)
   - Comparison tables
   - Visual examples
   - Key metrics (mAP, Recall, F2)

4. **Challenges & Solutions** (1-2 slides)
   - Technical issues (MPS compatibility, memory)
   - Solutions implemented
   - Lessons learned

5. **Conclusion** (1 slide)
   - Best performance achieved
   - Future improvements
   - Recommendations

### File Organization

```
reports/
├── baseline_results.md           # Phase 1 analysis
├── phase2_analysis.md            # Phase 2 analysis
├── final_report/
│   ├── methodology.md
│   ├── results.md
│   ├── analysis.md
│   └── figures/
│       ├── training_curves.png
│       ├── performance_comparison.png
│       ├── confusion_matrix.png
│       └── example_predictions.png
└── presentation/
    ├── slides.pptx (or .pdf)
    └── figures/
```

### Best Practices

- Generate figures after each experiment
- Keep analysis notes in markdown files
- Export key metrics to CSV for easy table generation
- Save high-resolution images (300+ DPI) for publication
- Document decisions and rationale in reports
- Include code snippets for reproducibility

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
uv run src/training/train.py

# Evaluate
uv run src/evaluation/f2_score.py

# Predict
uv run src/inference/predict.py

# Ablation study
uv run src/training/ablation_study.py --fold 0 --epochs 5 --device mps

# Compare ablation results
uv run src/evaluation/compare_ablation.py --fold 0

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
