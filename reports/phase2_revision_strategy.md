# Phase 2 Revision Strategy

## Executive Summary

Phase 2 optimization failed due to removing HSV augmentation, which is critical for underwater imagery. This revision takes an **incremental approach** to safely test improvements on the M4 Max MPS backend.

## What Went Wrong in Original Phase 2

| Issue | Impact | Root Cause |
|-------|--------|------------|
| Removed HSV augmentation | -47% mAP50 | Lost color generalization for underwater images |
| Aggressive box reduction (7.5→5.0) | Unknown | Too large a change to isolate effect |
| Rotation disabled | Limited augmentation | MPS TAL compatibility issue |
| OOM crashes | Training instability | MPS memory leak in long runs |

**Key Learning**: Don't change multiple things at once - can't isolate what hurt performance.

## Revised Strategy: Incremental Testing

### Test 1 (Current): Conservative Optimization
**Script**: `src/training/train_phase2_revised.py`

**Changes from baseline**:
- ✅ **KEEP** HSV augmentation (hsv_h=0.015, hsv_s=0.7, hsv_v=0.4)
- ⚠️ **DISABLE** rotation (degrees=0.0) - avoid TAL errors
- 📊 **SMALL** box weight change: 7.5 → 6.5 (not 5.0)
- ⏱️ **SHORTER** training: 20 epochs (test stability)

**Hypothesis**: Small box reduction + HSV augmentation should maintain or improve baseline performance.

**Expected Outcome**:
- Best case: mAP50 > 0.154 (beat baseline)
- Acceptable: mAP50 ≈ 0.154 (match baseline)
- Failure: mAP50 < 0.154 (worse than baseline)

### Test 2 (If Test 1 Succeeds): Add More Epochs
- Same settings as Test 1
- Increase to 30-40 epochs
- Monitor for OOM/stability issues

### Test 3 (If Test 2 Succeeds): Further Box Reduction
- Keep HSV enabled
- Try box=6.0 or box=5.5
- 30 epochs

### Alternative Path: SAHI on Baseline
If incremental changes don't improve over baseline:
- **Option**: Apply SAHI (Slicing Aided Hyper Inference) to baseline model
- **Advantage**: No retraining needed, works on existing weights
- **Expected gain**: +5-8% recall (better small object detection)

## Comparison: Baseline vs Phase 2 vs Phase 2 Revised

| Configuration | HSV Aug | Rotation | Box | Epochs | mAP50 | Status |
|--------------|---------|----------|-----|--------|-------|--------|
| **Baseline** | ✅ Enabled | ❌ Disabled | 7.5 | 10 | **0.154** | ✅ Best |
| **Phase 2** | ❌ Disabled | ❌ Disabled | 5.0 | 15 | 0.082 | ⚠️ Failed |
| **Phase 2 Rev** | ✅ Enabled | ❌ Disabled | 6.5 | 20 | TBD | 🔄 Testing |

## Why This Approach

### Advantages
1. **Incremental**: Change one variable at a time - can isolate effects
2. **Safe**: Keeps what worked (HSV augmentation)
3. **Compatible**: Works with M4 Max MPS backend
4. **Fast**: 20 epochs ≈ 3-4 hours (vs 12+ for 50 epochs)

### MPS Backend Constraints
- ✅ HSV augmentation: Compatible
- ✅ Horizontal flips: Compatible
- ✅ Mosaic: Compatible
- ⚠️ Rotation: **Incompatible** (causes TAL errors)
- ⚠️ Mixup: **Incompatible** (causes TAL errors)

### Winning Solution Adaptations
The 1st place Kaggle solution used:
- ✅ box=0.2 (we're testing 6.5, moving toward this)
- ✅ Rotation augmentation (we **can't** use on MPS)
- ❌ NO HSV augmentation (we **should** use for our underwater data)

**Our adaptation**: Keep HSV (helps our case), skip rotation (MPS limitation), gradually reduce box weight.

## Success Criteria

### Minimum Viable Success
- mAP50 ≥ 0.154 (match or beat baseline)
- Training completes without OOM crashes
- Recall improves (> 9.1%)

### Ideal Success
- mAP50 > 0.18 (+17% over baseline)
- Recall > 12% (+32% over baseline)
- Stable training (consistent epoch times)

## Next Steps After Revision

If Phase 2 Revised succeeds:
1. **Phase 3**: Temporal post-processing (attention area boosting)
2. **Phase 4**: Train final model on all videos, create submission
3. **Optional**: SAHI inference to boost small object detection

If Phase 2 Revised fails to beat baseline:
1. Skip further training optimization
2. Apply SAHI to baseline model (no retraining)
3. Move directly to Phase 3 (temporal post-processing)
4. Focus on inference-time improvements

## Training Command

```bash
# Test Phase 2 Revised on Fold 0 (20 epochs)
python src/training/train_phase2_revised.py --fold 0 --epochs 20

# If successful, extend training
python src/training/train_phase2_revised.py --fold 0 --epochs 30

# Train all folds (if validated)
for fold in 0 1 2; do
    python src/training/train_phase2_revised.py --fold $fold --epochs 30
done
```

## Timeline

- **Phase 2 Revised (Test 1)**: 20 epochs × ~10min/epoch = 3-4 hours
- **Analysis**: 1 hour (compare results, generate figures)
- **Decision point**: Continue optimization or pivot to SAHI

---

**Created**: 2025-12-22
**Status**: Ready for testing
**Approach**: Incremental, evidence-based optimization
