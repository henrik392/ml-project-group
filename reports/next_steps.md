# Phase 2 Completion: Next Steps & Experiments

**Date**: 2025-12-22 (Updated with YOLOv11 research)
**Current Status**: Ablation study complete + YOLOv11 vs YOLOv5 analysis complete
**Current Best**: Baseline @ 10 epochs, mAP50 = 0.154, Recall = 0.091, CV F2 ≈ 0.11

## 🔄 Major Update: YOLOv11 Research Changes Priorities

**Key realization**: The competition used YOLOv5 (2022). We're using YOLOv11 (2024) - fundamentally different architecture.

**What changed**:
1. **YOLOv11 defaults are strong** → Our ablation study validated this!
2. **Resolution matters most** → 1280px is now TOP priority (biggest small-object win)
3. **Temporal smoothing has very high ROI** → Moved from Phase 3 to Priority 2
4. **SAHI is expensive** → Downgraded to "only if needed" (Priority 5)
5. **Don't blindly copy YOLOv5 hyperparameters** → Focus on architecture not tuning

See `reports/yolov11_vs_yolov5_analysis.md` for full analysis.

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

**UPDATED BASED ON YOLOV11 RESEARCH** (see reports/yolov11_vs_yolov5_analysis.md)

Key insight: YOLOv11 defaults are strong. Focus on **resolution + temporal** over hyperparameter tuning.

### 🔥 Priority 1: Resolution Increase (BIGGEST WIN FOR SMALL OBJECTS)

**Rationale**: Research shows this is the #1 improvement for tiny objects in YOLOv11

**Experiment 1.1: 1280px Resolution @ 30 Epochs** ⭐ TOP PRIORITY
```bash
uv run src/training/train_baseline.py --fold 0 --epochs 30 --imgsz 1280 --model n --device mps --batch 4
```

**Expected outcome**:
- mAP50: 0.25-0.35 (+60-130% vs 640px)
- Recall: 0.25-0.35 (+175-285% vs 640px)
- Small objects: +30-50% detection rate
- Time: ~12 hours (4× slower due to 4× pixels)
- Memory: Requires batch=4 (not 16)

**Success criteria**: Recall > 0.25 OR mAP50 > 0.25

**Experiment 1.2: Baseline 640px @ 30 Epochs** (for comparison)
```bash
uv run src/training/train_baseline.py --fold 0 --epochs 30 --model n --device mps
```

**Expected outcome**:
- mAP50: 0.20-0.25
- Recall: 0.15-0.20
- Time: ~3 hours

**Success criteria**: Establishes baseline for resolution comparison

---

### 🎯 Priority 2: Temporal Post-Processing (VERY HIGH ROI)

**Rationale**: Research shows this has **very high ROI** - 1st place secret sauce, minimal cost

**Experiment 2.1: Temporal Confidence Boosting**
```bash
# Create src/postprocessing/temporal_smoothing.py
# Boost confidence for detections in nearby frames
```

**Strategy**:
- If starfish detected in frame N
- Boost confidence scores in frames N-2, N-1, N+1, N+2
- Use video sequence information (starfish move slowly)
- Create "attention areas" around previous detections

**Expected outcome**:
- F2: +5-10% (0.40 → 0.45-0.50)
- Recall: +10-20% via confidence boosting
- Precision: Minimal impact or slight improvement
- Cost: Minimal (post-processing only, no retraining)

**Success criteria**: F2 > 0.45 on Fold 0

**Implementation steps**:
1. Extract frame sequences from videos
2. Track detection positions across frames
3. Implement confidence boosting logic
4. Test on validation set
5. Tune boosting parameters (radius, strength)

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

### ⚡ Priority 5: SAHI Slicing (ONLY IF STILL RECALL-LIMITED)

**Rationale**: Research shows SAHI is expensive (2-6× slower). Use only if above methods insufficient.

**Experiment 5.1: SAHI Implementation**
```bash
# Create src/inference/sahi_predict.py
# Use best model weights from Priorities 1-4
```

**When to run**: Only if resolution + temporal + threshold gives F2 < 0.60

**SAHI configuration**:
- Slice size: 640×640
- Overlap ratio: 0.3-0.4
- Confidence threshold: Optimized from Priority 3

**Expected outcome**:
- Recall: +50-100% (but 2-6× slower)
- F2: +5-15%
- Cost: Very expensive (use selectively)

**Success criteria**: F2 > 0.65

---

### ⚡ Priority 6: Larger Model (IF NEEDED)

**Rationale**: More model capacity for small objects

**Experiment 6.1: YOLOv11s @ 30 Epochs (1280px)**
```bash
uv run src/training/train_baseline.py --fold 0 --epochs 30 --model s --imgsz 1280 --batch 2
```

**When to run**: Only if 1280px YOLOv11n + temporal + threshold gives F2 < 0.65

**Expected outcome**:
- mAP50: +5-10% vs nano
- Recall: +3-5% vs nano
- Time: ~20-24 hours (3× slower than nano @ 1280px)

**Trade-off**: Better performance but much slower training/inference

---

## Decision Tree (Updated for YOLOv11)

```
START
  │
  ├─> Train 1280px @ 30 epochs (Fold 0) 🔥 PRIORITY 1
  │     │
  │     ├─> Recall > 0.25? ─YES─> ✅ Excellent, continue
  │     │                   NO──> ⚠️ Try 640px comparison, may need YOLOv11s
  │     │
  │     └─> Implement temporal smoothing 🎯 PRIORITY 2
  │           │
  │           ├─> F2 > 0.45? ─YES─> ✅ Great progress!
  │           │                NO──> ⚠️ Check temporal logic
  │           │
  │           └─> Optimize confidence threshold 📊 PRIORITY 3
  │                 │
  │                 ├─> F2 > 0.55? ─YES─> ✅ Near target!
  │                 │                NO──> ⚠️ May need SAHI or larger model
  │                 │
  │                 └─> Train Fold 1 & 2 (1280px + temporal)
  │                       │
  │                       ├─> Mean CV F2 > 0.60? ─YES─> ✅ SUCCESS!
  │                       │                        NO──> Try SAHI or YOLOv11s
  │                       │
  │                       └─> If still F2 < 0.60:
  │                             ├─> Try SAHI (Priority 5)
  │                             └─> Try YOLOv11s @ 1280px (Priority 6)
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

## Immediate Actions (Updated Based on Research)

### Option A: Start with Resolution (Recommended 🔥)

**Best for**: Maximum small-object improvement

```bash
# TOP PRIORITY: 1280px resolution
uv run src/training/train_baseline.py --fold 0 --epochs 30 --imgsz 1280 --batch 4 --device mps
```

**Time**: ~12 hours
**Expected**: Recall +175-285% vs current

### Option B: Establish 640px Baseline First (Conservative)

**Best for**: Fair comparison between resolutions

```bash
# Baseline at 640px for comparison
uv run src/training/train_baseline.py --fold 0 --epochs 30 --imgsz 640 --device mps
```

**Time**: ~3 hours
**Expected**: mAP50 = 0.20-0.25

**Then run Option A after completion**

### While Training

1. **Prepare temporal smoothing**: Implement confidence boosting logic
2. **Study SAHI**: Understand implementation (but deprioritized now)
3. **Monitor training**: Check GPU usage, memory, convergence

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
