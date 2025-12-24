# Phase 2: Optimized Detection - Training Analysis

## Executive Summary

**Training Status**: Incomplete (15/15 epochs completed, crashed due to OOM)
**Date**: 2025-12-19
**Model**: YOLOv11n (2.59M parameters)
**Device**: M4 Max (MPS backend)
**Training Time**: ~12 hours

### Key Findings

⚠️ **Phase 2 performance degraded compared to baseline**
- mAP50 decreased from 0.154 → 0.082 (-47%)
- Recall decreased from 9.1% → 5.2% (-43%)
- Precision decreased from 62.0% → 49.9% (-19%)

**Root Causes Identified**:
1. Aggressive hyperparameter changes (box=5.0) may have destabilized training
2. Removal of augmentations (rotation, mixup) due to MPS compatibility issues
3. Memory issues causing training crashes (exit code 137 - OOM)

---

## Training Configuration

### Phase 2 Settings

```yaml
Model: YOLOv11n
Epochs: 15 (target), 15 (actual before crash)
Batch Size: 16
Image Size: 640x640
Device: MPS (M4 Max)
Optimizer: AdamW (lr=0.002, momentum=0.9)

Hyperparameters:
- box: 5.0 (reduced from default 7.5)
- cls: 0.5 (default)
- dfl: 1.5 (default)

Augmentations:
- degrees: 0.0 (disabled due to TAL errors)
- mixup: 0.0 (disabled due to TAL errors)
- fliplr: 0.5 (horizontal flips)
- hsv_h/s/v: 0.0 (disabled per winning solution)
- mosaic: 1.0 (enabled)
- translate: 0.1
- scale: 0.5
```

### Baseline (Phase 1) Settings

```yaml
Model: YOLOv11n
Epochs: 10
Batch Size: 16
Image Size: 640x640
Device: MPS (M4 Max)
Optimizer: AdamW (lr=0.002, momentum=0.9)

Hyperparameters:
- All defaults (box=7.5, cls=0.5, dfl=1.5)

Augmentations:
- All defaults (standard YOLO augmentations)
```

---

## Performance Comparison

### Final Metrics (Epoch 15 vs Baseline Epoch 10)

| Metric | Phase 2 (Epoch 15) | Baseline (Epoch 10) | Change | Status |
|--------|-------------------|---------------------|--------|--------|
| **mAP50** | 0.082 (8.2%) | 0.154 (15.4%) | -47% | ⚠️ Worse |
| **mAP50-95** | 0.040 (4.0%) | 0.078 (7.8%) | -49% | ⚠️ Worse |
| **Precision** | 0.499 (49.9%) | 0.620 (62.0%) | -19% | ⚠️ Worse |
| **Recall** | 0.052 (5.2%) | 0.091 (9.1%) | -43% | ⚠️ Worse |

### Training Progression (Phase 2)

| Epoch | mAP50 | mAP50-95 | Precision | Recall | Notes |
|-------|-------|----------|-----------|--------|-------|
| 1 | 0.042 | 0.018 | 0.131 | 0.064 | Initial |
| 3 | 0.052 | 0.020 | 0.471 | 0.045 | High precision spike |
| 5 | 0.057 | 0.023 | 0.244 | 0.064 | Unstable |
| 9 | 0.086 | 0.042 | 0.085 | 0.089 | Best recall |
| 12 | 0.086 | 0.045 | 0.100 | 0.064 | Best mAP50-95 |
| 15 | 0.082 | 0.040 | 0.499 | 0.052 | Final (OOM crash) |

**Best Performance**: Epoch 12 (mAP50-95: 0.045)

---

## Loss Analysis

### Final Training Losses (Epoch 15)

| Loss Type | Value | vs Baseline | Interpretation |
|-----------|-------|-------------|----------------|
| Box Loss | 1.030 | -40% ✅ | Better localization |
| Classification Loss | 0.929 | -48% ✅ | Better confidence |
| DFL Loss | 0.950 | -6% ✅ | Slightly better box quality |

### Validation Losses (Epoch 15)

| Loss Type | Value | vs Baseline | Interpretation |
|-----------|-------|-------------|----------------|
| Box Loss | 1.305 | -34% ✅ | Generalization improving |
| Classification Loss | 5.139 | +36% ⚠️ | Overfitting on classification |
| DFL Loss | 1.057 | +3% ≈ | Similar box quality |

**Key Observation**: Training losses improved significantly, but validation classification loss increased, suggesting overfitting or difficulty learning the single-class problem.

---

## Technical Issues Encountered

### 1. Task-Aligned Assigner (TAL) Errors

**Error**: RuntimeError: shape mismatch in TAL forward pass

**Attempted Solutions**:
- ❌ Reduced iou parameter (0.3 → default 0.7)
- ❌ Reduced box parameter (0.2 → 5.0)
- ❌ Disabled mixup augmentation
- ✅ Disabled rotation augmentation (degrees=0.0)

**Root Cause**: Combination of rotation + mosaic augmentation triggers shape mismatches in YOLOv11's Task-Aligned Assigner on MPS backend.

### 2. Out of Memory (OOM) Crashes

**Error**: Exit code 137 (SIGKILL - memory limit exceeded)

**Observations**:
- Training completed 15 epochs before crashing
- GPU memory usage: ~4.3-5.4GB (spiked at epoch 6)
- Training time drastically increased after epoch 5 (979s → 39,322s per epoch)

**Likely Causes**:
- Memory leak in MPS backend
- Excessive caching/logging during training
- Large batch size for extended training

### 3. Training Instability

**Evidence**:
- Precision fluctuated wildly: 13% → 47% → 24% → 50%
- Recall remained consistently low (4-9%)
- mAP50 plateaued around epoch 9-12

---

## Analysis & Recommendations

### What Went Wrong

1. **Removed Critical Augmentations**
   - Rotation augmentation (degrees=15) is important for underwater imagery
   - Mixup augmentation helps with small object detection
   - Both were disabled due to MPS compatibility issues

2. **Hyperparameter Tuning Too Conservative**
   - box=5.0 is closer to default (7.5) than target (0.2)
   - No significant benefit from this change
   - May have reduced model's sensitivity to small objects

3. **MPS Backend Limitations**
   - Task-Aligned Assigner incompatibility
   - Memory management issues
   - Training instability at extended durations

### Performance Degradation Explanation

Despite **better training losses**, validation metrics worsened because:

1. **Overfitting**: Classification loss on validation increased (5.14 vs 3.78 baseline)
2. **Loss of Generalization**: Removed augmentations reduced model's ability to generalize
3. **Small Object Detection**: box=5.0 didn't improve small starfish detection as intended

### Path Forward

#### Option A: Return to Baseline + Incremental Improvements (Recommended)

Use baseline configuration as foundation:
1. Start with proven baseline settings
2. Add augmentations one-by-one (test stability)
3. Use CPU/CUDA for training if available (avoid MPS issues)
4. Increase epochs to 30-50 for proper convergence

#### Option B: Switch to Different Backend

Train on:
- Google Colab (free CUDA GPUs)
- Kaggle Kernels (free T4/P100 GPUs)
- AWS/GCP cloud instances

This would enable:
- Full augmentation suite (rotation, mixup, etc.)
- Aggressive hyperparameters (box=0.2)
- Stable training without TAL errors

#### Option C: Alternative Approaches

1. **SAHI (Slicing Aided Hyper Inference)**
   - Use baseline model + SAHI for small object detection
   - No retraining needed
   - Apply during inference only

2. **Test-Time Augmentation (TTA)**
   - Multiple inference passes with augmentations
   - Average predictions
   - Improve recall without retraining

---

## Deliverables

### Generated Assets

**Location**: `runs/optimized/yolo11n_fold0_opt7/`

1. **Training Curves**: `results.png`
   - Loss progression over 15 epochs
   - Metric trends

2. **Performance Curves**:
   - `BoxF1_curve.png` - F1 score vs confidence threshold
   - `BoxP_curve.png` - Precision vs confidence threshold
   - `BoxR_curve.png` - Recall vs confidence threshold
   - `BoxPR_curve.png` - Precision-Recall curve

3. **Confusion Matrix**:
   - `confusion_matrix.png` - Absolute counts
   - `confusion_matrix_normalized.png` - Normalized

4. **Predictions**:
   - `val_batch0_pred.jpg` - Validation predictions
   - `val_batch1_pred.jpg`
   - `val_batch2_pred.jpg`

5. **Training Data**: `results.csv` - Full metrics CSV

6. **Model Weights**: `weights/last.pt` - Final checkpoint

---

## Lessons Learned

### Technical

1. **MPS Backend Compatibility**: YOLOv11 + aggressive augmentations + MPS = instability
2. **Hyperparameter Sensitivity**: Small changes can have large negative impacts
3. **Augmentation Importance**: Data augmentation critical for generalization
4. **Memory Management**: Long training runs on MPS prone to memory leaks

### Methodological

1. **Incremental Changes**: Test one change at a time, not multiple simultaneously
2. **Validation Strategy**: More frequent validation helps catch issues earlier
3. **Baseline Preservation**: Always maintain working baseline for comparison
4. **Platform Selection**: Choose training platform based on compatibility, not just availability

---

## Next Steps

### Immediate Actions (Phase 2 Revision)

1. ✅ Document Phase 2 findings
2. ⏳ Decide on training platform (MPS vs cloud)
3. ⏳ Design Phase 2 revision with incremental changes
4. ⏳ Run ablation study: baseline + one augmentation at a time

### Future Phases

**Phase 3**: Temporal post-processing (if baseline is stable)
**Phase 4**: Final model training on all folds
**Phase 5**: SAHI + TTA optimization

---

## Conclusion

Phase 2 demonstrated that **careful hyperparameter tuning and platform selection are critical**. The aggressive changes made to work around MPS limitations resulted in worse performance than the baseline.

**Recommendation**: Revert to baseline configuration and either:
1. Use cloud GPUs for stable training with full augmentations
2. Apply post-processing techniques (SAHI, TTA) to baseline model
3. Use ensemble of multiple baseline models trained with different seeds

The baseline model (mAP50: 0.154, Recall: 9.1%) is currently our best performer and should be the foundation for future improvements.

---

**Report Generated**: 2025-12-19
**Training Duration**: 12.08 hours (43,908 seconds)
**Total Epochs**: 15 (incomplete due to OOM)
