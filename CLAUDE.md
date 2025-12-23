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

## Scientific Rigor & Academic Standards

**CRITICAL: Distinguish Facts from Hypotheses**

When documenting findings, ALWAYS differentiate between:

### ✅ Facts (Empirically Verified)
- Statements supported by experimental data from this project
- Observable results from ablation studies or experiments
- Quantitative measurements (mAP50 = 0.154, Recall = 9.1%)
- Technical errors with stack traces or logs

**Language to use:**
- "Our experiments show..."
- "The ablation study demonstrates..."
- "We observed that..."
- "The results indicate..."
- "Data shows..."

### ⚠️ Hypotheses (Unverified Assumptions)
- Explanations for observed phenomena without direct testing
- Inferences about causation without controlled experiments
- Assumptions about underlying mechanisms
- Predictions about untested scenarios

**Language to use:**
- "We hypothesize that..."
- "This suggests..."
- "Possible explanation: ..."
- "One potential reason could be..."
- "This may indicate..."
- "Further investigation is needed to confirm..."

### Examples

❌ **Wrong (stating hypothesis as fact):**
> "Rotation hurts performance because COTS have consistent orientations in video frames."

✅ **Correct (properly framed as hypothesis):**
> "Rotation degrades performance (-12.7% mAP50). **Hypothesis**: This may be due to COTS having preferred orientations in underwater video frames, though this requires verification through ground truth orientation analysis."

❌ **Wrong:**
> "The model overfits because the dataset is too small."

✅ **Correct:**
> "The model shows signs of overfitting (training loss decreases while validation loss increases). **Possible explanations** include limited dataset size (23,501 images) or insufficient regularization, but controlled experiments are needed to isolate the root cause."

### Report Structure

Every analysis section should include:

1. **Observation** (fact): What we measured
2. **Hypothesis** (clearly labeled): Why we think it happened
3. **Evidence** (if available): Supporting data for the hypothesis
4. **Future Work** (if applicable): How to test the hypothesis

**Example:**

```markdown
### Finding: HSV Augmentation Impact

**Observation**: Disabling HSV augmentation resulted in -34.3% mAP50 degradation
(from 0.126 to 0.083).

**Hypothesis**: HSV augmentation is critical for underwater imagery because:
1. Underwater scenes exhibit extreme color variation due to depth and lighting
2. HSV augmentation helps the model generalize across these conditions
3. Without it, the model may overfit to specific color distributions in the
   training set

**Evidence**:
- Validation box loss increased from 2.10 to 2.36 (poor localization)
- Training classification loss decreased (1.71 vs 2.09), but validation
  classification loss also decreased (4.35 vs 5.20), suggesting the model
  learned simpler patterns

**Further Investigation Needed**:
- Test HSV augmentation at different intensities (hsv_h=0.005, 0.01, 0.02)
- Analyze color distribution differences between training and validation sets
- Visualize learned features with/without HSV augmentation
```

### Academic Writing Standards

- Use precise, qualified language
- Avoid absolute statements without evidence
- Acknowledge limitations and uncertainties
- Separate correlation from causation
- Document assumptions explicitly
- Provide error bars and confidence intervals when possible

This ensures our reports and presentations maintain academic integrity and
clearly communicate the certainty level of our findings.

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
