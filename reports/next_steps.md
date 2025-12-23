# Phase 2 Completion: Next Steps & Experiments

**Date**: 2025-12-22
**Current Status**: Ablation study complete, ready for next phase
**Current Best**: Baseline @ 10 epochs, mAP50 = 0.154, Recall = 0.091, CV F2 ≈ 0.11

---

## Critical Insight: Recall is the Bottleneck

```
Target:     CV F2 > 0.70
Current:    CV F2 ≈ 0.11 (estimated from mAP50/recall)

Bottleneck: Recall = 9.1% (missing 91% of starfish!)
Precision:  62.0% (good, but we need more detections)
```

**F2 Formula**: F2 = 5 × (P × R) / (4×P + R)
- F2 heavily weights recall (β=2)
- To reach F2 > 0.70, we need R > 0.30 (3× current!)

---

## Recommended Experiments (Priority Order)

### 🔥 Priority 1: Extend Training (MUST DO)

**Hypothesis**: 10 epochs is insufficient, model hasn't converged

**Experiment 1.1: Baseline @ 30 Epochs**
```bash
uv run src/training/train_baseline.py --fold 0 --epochs 30 --model n --device mps
```

**Expected outcome**:
- mAP50: 0.20-0.25 (+30-60% improvement)
- Recall: 0.15-0.20 (+65-120% improvement)
- Time: ~3 hours

**Success criteria**: mAP50 > 0.18 OR Recall > 0.15

**Experiment 1.2: Baseline + Mixup @ 30 Epochs** (parallel)
```bash
# Modify train_baseline.py to add mixup=0.1
uv run src/training/train_baseline.py --fold 0 --epochs 30 --model n --device mps
```

**Expected outcome**:
- mAP50: 0.20-0.26 (similar or +2-3% vs baseline)
- Recall: 0.15-0.21
- Time: ~3 hours

**Success criteria**: mAP50 > baseline_30epochs

---

### 🎯 Priority 2: SAHI Inference (HIGH IMPACT)

**Hypothesis**: Slicing images will dramatically improve small object detection

**Experiment 2.1: SAHI Implementation**
```bash
# Create src/inference/sahi_predict.py
# Use baseline model weights from Fold 0
```

**SAHI configuration to test**:
- Slice size: 640×640 (same as training)
- Overlap ratio: 0.2, 0.3, 0.4 (test all)
- Confidence threshold: 0.25 (default)

**Expected outcome**:
- Recall: +50-100% (0.09 → 0.13-0.18 @ 10 epochs)
- Recall: +50-100% (0.15 → 0.22-0.30 @ 30 epochs)
- mAP50: May decrease slightly (more false positives)

**Success criteria**: Recall > 0.20 on Fold 0 validation

**Implementation steps**:
1. Install SAHI: `uv add sahi`
2. Create prediction script with slicing
3. Test on 100 validation images
4. Sweep overlap ratios
5. Run on full Fold 0 validation set

---

### 📊 Priority 3: Confidence Threshold Optimization

**Hypothesis**: Lower threshold = more detections = higher recall

**Experiment 3.1: Threshold Sweep**
```bash
# Create src/evaluation/optimize_threshold.py
# Test thresholds: 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5
```

**Test on**:
- Baseline @ 30 epochs predictions
- Baseline + SAHI predictions

**Metrics to track**:
- Precision, Recall, F2 for each threshold
- Find threshold that maximizes F2

**Expected outcome**:
- Optimal threshold: 0.10-0.20 (lower than default 0.25)
- F2 boost: +20-40%

**Success criteria**: F2 > 0.40 on Fold 0

---

### 🔬 Priority 4: 3-Fold Cross-Validation (VALIDATION)

**Hypothesis**: Fold 0 results generalize to all folds

**Experiment 4.1: Train All Folds**
```bash
# After validating baseline @ 30 epochs on Fold 0
uv run src/training/train_baseline.py --fold 1 --epochs 30 --model n --device mps
uv run src/training/train_baseline.py --fold 2 --epochs 30 --model n --device mps
```

**Time**: ~6 hours (2 folds × 3 hours)

**Expected outcome**:
- Mean CV mAP50: 0.18-0.24
- Mean CV F2 (with SAHI + threshold tuning): 0.50-0.65

**Success criteria**: Mean CV F2 > 0.50

---

### ⚡ Priority 5: Larger Model (IF NEEDED)

**Hypothesis**: YOLOv11s has more capacity for small objects

**Experiment 5.1: YOLOv11s @ 30 Epochs**
```bash
uv run src/training/train_baseline.py --fold 0 --epochs 30 --model s --device mps
```

**When to run**: Only if baseline @ 30 epochs + SAHI + threshold tuning gives CV F2 < 0.60

**Expected outcome**:
- mAP50: +5-10% vs nano
- Recall: +3-5% vs nano
- Time: ~4-5 hours (slower than nano)

**Trade-off**: Better performance but slower inference

---

## Decision Tree

```
START
  │
  ├─> Train baseline @ 30 epochs (Fold 0)
  │     │
  │     ├─> mAP50 > 0.20? ─YES─> ✅ Continue
  │     │                  NO──> ⚠️ Try YOLOv11s
  │     │
  │     └─> Implement SAHI
  │           │
  │           ├─> Recall > 0.20? ─YES─> ✅ Continue
  │           │                   NO──> ⚠️ Try larger slices or YOLOv11s
  │           │
  │           └─> Optimize confidence threshold
  │                 │
  │                 ├─> F2 > 0.40? ─YES─> ✅ Continue
  │                 │                NO──> ⚠️ Need Phase 3 (temporal)
  │                 │
  │                 └─> Train Fold 1 & 2
  │                       │
  │                       ├─> Mean CV F2 > 0.50? ─YES─> ✅ Move to Phase 3
  │                       │                        NO──> ⚠️ Try YOLOv11s or adjust strategy
  │                       │
  │                       └─> Phase 3: Temporal Post-Processing
  │                             │
  │                             └─> Target: CV F2 > 0.70
```

---

## Timeline Estimate

**Week 1**: Core improvements
- Day 1: Baseline @ 30 epochs (Fold 0) + analysis
- Day 2: SAHI implementation + testing
- Day 3: Threshold optimization + F2 calculation
- Day 4: Train Fold 1 & 2 @ 30 epochs
- Day 5: Calculate 3-fold CV, generate report

**Week 2**: Polish + Phase 3
- Day 1-2: Temporal post-processing (attention area boosting)
- Day 3: Final model training on all videos
- Day 4: Submission generation
- Day 5: Final report + presentation

---

## Success Metrics

### Minimum Viable Success (Pass threshold)
- ✅ CV F2 > 0.50
- ✅ Recall > 0.25 (on any fold)
- ✅ Ablation study documented
- ✅ Phase 2 complete with learnings

### Target Success (Project goal)
- ✅ CV F2 > 0.70
- ✅ Recall > 0.35
- ✅ All phases completed
- ✅ Kaggle submission made

### Stretch Success (Competitive)
- ✅ CV F2 > 0.75
- ✅ Recall > 0.45
- ✅ Ensemble models
- ✅ Multiple submissions

---

## Risk Assessment

### Low Risk (High confidence)
1. ✅ Baseline @ 30 epochs will improve over 10 epochs
2. ✅ SAHI will boost recall significantly
3. ✅ Threshold tuning will optimize F2

### Medium Risk (Moderate confidence)
4. ⚠️ Reaching CV F2 > 0.60 with nano model alone
5. ⚠️ 3-fold results matching Fold 0 performance

### High Risk (Low confidence)
6. ⚠️ Reaching CV F2 > 0.70 without temporal post-processing
7. ⚠️ YOLOv11s fitting in memory with batch=16 on M4 Max

---

## Immediate Actions (Next 24 Hours)

1. **Start training**: Launch baseline @ 30 epochs on Fold 0
   ```bash
   uv run src/training/train_baseline.py --fold 0 --epochs 30 --model n --device mps
   ```

2. **Monitor progress**: Check after 1 hour to ensure no crashes

3. **Prepare SAHI**: While training runs, implement SAHI inference script

4. **Document setup**: Create training log template for tracking experiments

---

## Questions to Answer

### High Priority
1. **Does baseline converge by 30 epochs?** → Experiment 1.1
2. **How much does SAHI improve recall?** → Experiment 2.1
3. **What's the optimal confidence threshold?** → Experiment 3.1

### Medium Priority
4. **Does mixup help at 30 epochs?** → Experiment 1.2
5. **Do we need YOLOv11s?** → Depends on Experiment 1.1 + 2.1 results
6. **What's the true 3-fold CV F2?** → Experiment 4.1

### Low Priority
7. **Can we reach F2 > 0.70 with just Phase 2?** → Probably no, need Phase 3
8. **Should we ensemble multiple models?** → Only if time permits
9. **Is 1280px image size beneficial?** → Low priority, very slow

---

## Tools to Create

### Training Tools
- `src/training/train_baseline.py` ✅ (exists)
- `src/training/train_extended.py` (wrapper for 30+ epochs with logging)

### Inference Tools
- `src/inference/sahi_predict.py` ⏳ (to create)
- `src/inference/ensemble_predict.py` (if needed)

### Evaluation Tools
- `src/evaluation/optimize_threshold.py` ⏳ (to create)
- `src/evaluation/calculate_cv_f2.py` ⏳ (to create)
- `src/evaluation/f2_score.py` ✅ (exists, may need updates)

### Visualization Tools
- `src/visualization/plot_training_progress.py` ⏳ (for 30-epoch runs)
- `src/visualization/plot_detection_examples.py` (for report)

---

## Expected Deliverables

### End of Phase 2
1. ✅ Ablation study analysis (complete)
2. ⏳ Baseline @ 30 epochs results (3 folds)
3. ⏳ SAHI implementation + results
4. ⏳ Confidence threshold optimization
5. ⏳ 3-fold CV F2 score calculation
6. ⏳ Updated PLAN.md with Phase 2 completion
7. ⏳ Presentation figures for final report

### Documentation
- Training logs for all experiments
- Performance comparison tables
- Visualizations (training curves, detection examples)
- Technical report on findings

---

**Status**: Ready to proceed with Priority 1 experiments
**Next Action**: Start baseline @ 30 epochs training
**Decision Point**: After Experiment 1.1 completes (~3 hours)
