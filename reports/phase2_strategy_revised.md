# Phase 2 Strategy: Resolution + Model Scaling

**Date**: 2025-12-24
**Status**: Ready to Execute
**Based on**: Modern YOLOv11 best practices (2024), not outdated YOLOv5 techniques (2022)

## Key Insight

**YOLOv11 has fundamentally different architecture than YOLOv5:**
- Anchor-free detection → better small-object localization
- Decoupled heads → cleaner classification vs box regression
- Better multi-scale features → fewer missed tiny objects
- NMS-free training → fewer duplicate boxes

**Result**: YOLOv11 defaults are already strong. Focus on **scaling**, not hyperparameter tuning.

---

## Strategy Overview

### Primary Approach: Scale Resolution + Model Size

**Highest ROI for small object detection:**
1. **Resolution scaling**: 640px → 1280px (2-3× improvement expected)
2. **Model scaling**: YOLOv11n → YOLOv11-s or YOLOv11-m (+20-30% recall)
3. **Confidence threshold tuning**: Optimize for F2 metric (recall-focused)

### Why This Works

- **Small objects**: COTS are tiny in 1920×1080 frames
- **Higher resolution**: Provides more pixels per starfish
- **Bigger models**: More parameters to learn fine-grained features
- **Lower threshold**: F2 metric prioritizes recall 5× over precision

---

## Experimental Plan

### Experiment 1: Resolution Scaling (Baseline Model)

**Goal**: Measure impact of resolution alone

**Configuration**:
- Model: YOLOv11n (keep baseline)
- Resolution: 1280px (vs 640px baseline)
- Batch size: 4-8 (reduced due to memory)
- Epochs: 15 (with early stopping patience=5)
- Hyperparameters: **Keep defaults**

**Expected Results**:
- mAP50: 0.25-0.35 (+100-180% improvement)
- Recall: 0.25-0.35 (+120-200% improvement)

### Experiment 2: Model Scaling (Baseline Resolution)

**Goal**: Measure impact of model size alone

**Configuration**:
- Model: YOLOv11-s (9M params vs 2.59M)
- Resolution: 640px (keep baseline)
- Batch size: 16
- Epochs: 15
- Hyperparameters: **Keep defaults**

**Expected Results**:
- mAP50: 0.20-0.25 (+60-100% improvement)
- Recall: 0.20-0.28 (+75-145% improvement)

### Experiment 3: Combined Scaling (PRIMARY)

**Goal**: Best of both worlds

**Configuration**:
- Model: YOLOv11-s
- Resolution: 1280px
- Batch size: 4-8
- Epochs: 15-20
- Hyperparameters: **Keep defaults**

**Expected Results**:
- mAP50: 0.30-0.40 (+140-220% improvement)
- Recall: 0.40-0.50 (+250-340% improvement)
- F2 Score: 0.40-0.50 (before confidence tuning)

### Experiment 4: Confidence Threshold Tuning

**Goal**: Optimize for F2 metric

**Method**:
1. Use best model from Experiment 3
2. Generate predictions at very low threshold (conf=0.001)
3. Post-process to find optimal threshold for F2 score
4. Test thresholds: [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]

**Expected Results**:
- F2 Score: 0.50-0.60 (with optimal threshold)
- Recall: 0.50-0.65
- Precision: 0.30-0.40 (acceptable trade-off)

---

## Implementation Details

### Memory Optimization for 1280px

**Challenge**: 1280px requires 4× memory vs 640px

**Solutions**:
1. Reduce batch size from 16 → 4 or 8
2. Use gradient accumulation if needed
3. Monitor MPS memory usage
4. Fall back to CPU if MPS OOM (slower but works)

### Training Configuration

```python
# Experiment 3: YOLOv11-s @ 1280px
config = {
    "model": "yolo11s.pt",
    "data": "configs/dataset_fold_0.yaml",
    "epochs": 15,
    "imgsz": 1280,
    "batch": 8,  # Adjust based on memory
    "device": "mps",
    "patience": 5,
    # KEEP ALL OTHER DEFAULTS
}
```

### Computational Cost

**Baseline** (YOLOv11n @ 640px):
- ~500-900s per epoch
- Total: ~2.5 hours for 15 epochs

**Scaled** (YOLOv11-s @ 1280px):
- Estimated: ~1500-2500s per epoch (3-4× slower)
- Total: ~8-10 hours for 15 epochs
- **Still manageable on M4 Max**

---

## Success Criteria

### Phase 2 Targets

| Metric | Baseline | Phase 2 Target | Stretch Goal |
|--------|----------|----------------|--------------|
| mAP50 | 0.126 | 0.30 | 0.40 |
| Recall | 0.114 | 0.40 | 0.50 |
| Precision | 0.471 | 0.35 | 0.45 |
| F2 Score | ~0.15 | 0.50 | 0.60 |

### Decision Rules

**If targets achieved**:
- ✅ Move to Phase 3 (temporal smoothing)
- Skip hyperparameter tuning (defaults work!)

**If targets NOT achieved**:
- Try YOLOv11-m (20M params) @ 1280px
- Consider selective hyperparameter tuning:
  - box=0.2 (vs default 7.5)
  - iou=0.3 (vs default 0.7)
- SAHI as last resort (expensive)

---

## What NOT to Do

### Avoid These Common Pitfalls

❌ **Don't copy YOLOv5 hyperparameters blindly**
- YOLOv11 architecture is fundamentally different
- Defaults are already optimized

❌ **Don't over-optimize augmentations**
- YOLOv11 defaults are strong
- Fancy augmentations rarely help
- "If baseline beats experiments → trust it"

❌ **Don't disable HSV augmentation**
- That was YOLOv5-specific finding
- YOLOv11 handles color variation better

❌ **Don't spend time on heavy ensembling**
- 1 strong YOLOv11 model ≈ old multi-model ensembles
- Save ensembling for final phase (if needed)

---

## Timeline

### Week 1: Experiments 1-3 (Scaling)

**Day 1-2**: Experiment 1 (Resolution scaling)
- Train YOLOv11n @ 1280px on Fold 0
- Evaluate and document results

**Day 3-4**: Experiment 2 (Model scaling)
- Train YOLOv11-s @ 640px on Fold 0
- Evaluate and compare to Exp 1

**Day 5-7**: Experiment 3 (Combined scaling)
- Train YOLOv11-s @ 1280px on Fold 0
- Full 3-fold CV if results are promising
- Document findings

### Week 2: Confidence Tuning + 3-Fold CV

**Day 1-2**: Experiment 4 (Confidence threshold)
- Generate predictions at multiple thresholds
- Find optimal for F2 score
- Validate on all 3 folds

**Day 3-5**: 3-Fold Cross-Validation
- Train best configuration on all 3 folds
- Calculate mean F2 score
- Generate final Phase 2 report

**Day 6-7**: Documentation and Analysis
- Update reports with results
- Generate visualizations
- Plan Phase 3 (temporal smoothing)

---

## Deliverables

1. **Trained Models**:
   - YOLOv11n @ 1280px (Fold 0)
   - YOLOv11-s @ 640px (Fold 0)
   - YOLOv11-s @ 1280px (All 3 folds)

2. **Analysis Report**:
   - Performance comparison table
   - Training curves for each experiment
   - Ablation study: resolution vs model size
   - Confidence threshold analysis

3. **Best Model**:
   - 3-fold CV F2 score
   - Optimal confidence threshold
   - Ready for Phase 3 (temporal smoothing)

---

## Phase 3 Preview: Temporal Smoothing

**After Phase 2 completes**, add temporal logic:

1. **Track detections across frames**
   - If starfish detected in frame N, boost confidence in frames N±K
   - Simple sliding window (no heavy trackers)

2. **Expected impact**: +10-20% F2 score (very high ROI)

3. **Target**: F2 > 0.60-0.70

**This is where the real magic happens for video data!**

---

## Summary

### Core Philosophy

> "YOLOv11 defaults are strong. Scale up, don't over-optimize."

### Key Changes from Old Strategy

| Old (YOLOv5-based) | New (YOLOv11 best practices) |
|-------------------|------------------------------|
| Copy winner's hyperparams | Keep defaults, scale resolution |
| Disable HSV, add rotation | Keep default augmentations |
| Focus on augmentation tuning | Focus on resolution + model size |
| Heavy hyperparameter search | Minimal tuning, trust defaults |
| Multi-model ensembles | Single strong model first |

### Expected Outcomes

- **Phase 2**: F2 > 0.50 (scaling alone)
- **Phase 3**: F2 > 0.60 (+ temporal smoothing)
- **Phase 4**: F2 > 0.70 (final optimizations)

**Ready to execute!** 🚀
