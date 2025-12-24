# YOLOv11 vs YOLOv5: Why Our Ablation Study Validates Modern Defaults

**Date**: 2025-12-22
**Context**: Research into YOLOv11 improvements since 2022 competition

---

## Critical Realization: The Competition Used YOLOv5 (2022)

The TensorFlow Great Barrier Reef competition was held in **2022** using **YOLOv5**. The winning solutions were optimized for YOLOv5's limitations. We are using **YOLOv11 (2024)**, which has fundamentally different architecture.

**This explains why our ablation study found baseline defaults are optimal!**

---

## What Changed: YOLOv5 → YOLOv11

### Architectural Improvements

| Feature | YOLOv5 (2022) | YOLOv11 (2024) | Impact on COTS Detection |
|---------|---------------|----------------|--------------------------|
| **Detection paradigm** | Anchor-based | **Anchor-free** | Better small-object localization |
| **Head architecture** | Coupled head | **Decoupled heads** | Cleaner classification vs box regression |
| **Multi-scale features** | Basic FPN | **Enhanced multi-scale** | Fewer missed tiny objects |
| **NMS handling** | Post-processing NMS | **NMS-free training** (v10+) | Fewer duplicate boxes |
| **Default hyperparameters** | Needed tuning | **Pre-optimized** | Good out-of-box performance |

**Key takeaway**: **1 YOLOv11 model ≈ old multi-model YOLOv5 ensembles**

---

## Why 1st Place YOLOv5 Solution Doesn't Apply Directly

### What 1st Place Did (YOLOv5, 2022)

```python
# 1st place hyperparameters (YOLOv5)
box=0.2      # REDUCED from YOLOv5 default (7.5)
iou_t=0.3    # IoU training threshold
hsv_h=0.0    # DISABLED HSV augmentation
hsv_s=0.0
hsv_v=0.0
degrees=15   # Rotation enabled
mixup=0.1    # Mixup enabled
```

**Why they needed these changes:**
- YOLOv5 defaults weren't optimized for small objects
- YOLOv5 had weaker multi-scale feature extraction
- Aggressive augmentation compensated for model limitations
- Lower box weight helped because YOLOv5 struggled with precise localization

### What We Found (YOLOv11, 2024)

```python
# Our ablation study results
box=7.5      # DEFAULT is BEST (reducing it hurts -7% to -31%)
hsv_h=0.015  # ENABLED is CRITICAL (-34% when disabled)
hsv_s=0.7
hsv_v=0.4
degrees=0.0  # DISABLED is better (-12.7% when enabled)
mixup=0.0    # Neutral (+0.1% when enabled)
```

**Why YOLOv11 defaults are better:**
- Anchor-free detection handles small objects natively
- Decoupled heads already separate box/class learning
- Enhanced multi-scale features reduce need for aggressive augmentation
- Better default hyperparameters optimized on modern datasets

---

## What Actually Matters for YOLOv11

### ✅ HIGH IMPACT (Do These)

1. **Higher Image Resolution** 🔥
   - **Biggest win for small starfish**
   - Test: 640 → 1280px
   - Expected: +20-40% recall on tiny objects
   - Cost: 4× compute, 4× memory

2. **Temporal Smoothing** 🔥
   - **Very high ROI** (1st place secret sauce)
   - Boost confidence for detections in nearby frames
   - Expected: +5-10% F2 score
   - Cost: Minimal (post-processing only)

3. **Confidence Threshold Tuning**
   - Critical for F2 metric (recall-heavy)
   - Lower threshold → more detections → higher recall
   - Expected: +20-30% F2 score
   - Cost: None (just threshold selection)

4. **Model Size Scaling**
   - YOLOv11n → YOLOv11s/m/l
   - More capacity for small objects
   - Expected: +5-15% mAP50
   - Cost: 3-10× slower inference

### ⚠️ MEDIUM IMPACT (Consider If Needed)

5. **SAHI (Slicing)**
   - Works for tiny objects
   - Expensive: 2-6× slower inference
   - Use only if still recall-limited after 1-4
   - Expected: +50-100% recall (but slower)

6. **Light Ensembling**
   - 2 models max (YOLOv11 reduces need)
   - Diminishing returns vs YOLOv5 era
   - Expected: +2-5% mAP50
   - Cost: 2× inference time

### ❌ LOW IMPACT (Don't Over-Optimize)

7. **Fancy Augmentations**
   - YOLOv11 defaults usually win
   - Our ablation study confirms this
   - Stick with baseline HSV + flip + mosaic

8. **Heavy Ensembling**
   - 5+ models not needed anymore
   - YOLOv11 is already strong alone

9. **Full Object Trackers**
   - Minimal gain vs simple temporal logic
   - Complex, brittle, slow

---

## Our Ablation Study: Validation of YOLOv11 Defaults

### What We Proved

**Observation**: Baseline YOLOv11 defaults outperformed all single-parameter changes.

**Why this makes sense now**:
- YOLOv11 defaults are pre-optimized on modern datasets
- Anchor-free detection handles small objects well
- Decoupled heads reduce need for aggressive hyperparameter tuning
- Our underwater dataset benefits from standard HSV augmentation

**Evidence**:
| Change | Impact | Interpretation |
|--------|--------|----------------|
| Disable HSV | -34.3% | YOLOv11 defaults include good augmentation |
| Reduce box weight | -7% to -31% | YOLOv11 box weight already optimized |
| Add rotation | -12.7% | YOLOv11 doesn't need aggressive augmentation |
| Add mixup | +0.1% | Neutral, defaults are sufficient |

**Conclusion**: **Trust the defaults and focus on architecture/resolution/temporal instead of hyperparameter tuning.**

---

## Updated Priority List for COTS Detection

### Priority 1: Architecture & Resolution (Biggest Wins)

1. **Test 1280px resolution** (Top priority!)
   ```bash
   # Expected: +20-40% recall on small objects
   uv run src/training/train_baseline.py --fold 0 --epochs 30 --imgsz 1280
   ```

2. **Extend training to 30+ epochs**
   ```bash
   # Model hasn't converged at 10 epochs
   uv run src/training/train_baseline.py --fold 0 --epochs 30 --imgsz 640
   ```

3. **Test larger model (YOLOv11s/m)**
   ```bash
   # More capacity for small objects
   uv run src/training/train_baseline.py --fold 0 --epochs 30 --model s
   ```

### Priority 2: Temporal Post-Processing (Very High ROI)

4. **Implement temporal confidence boosting**
   - Boost detections that appear in consecutive frames
   - 1st place secret sauce
   - Minimal cost, high impact

5. **Optimize confidence threshold for F2**
   - Lower threshold for higher recall
   - F2 metric heavily weights recall

### Priority 3: SAHI (Only If Still Recall-Limited)

6. **Test SAHI slicing**
   - 2-6× slower, but may help tiny objects
   - Use only if above methods insufficient

### Priority 4: Optional Enhancements

7. **Light ensemble (2 models max)**
8. **Patch-trained secondary model**

---

## Revised Understanding: Why HSV Augmentation Matters

### YOLOv5 1st Place: "No HSV"

**Hypothesis**: Their dataset may have had:
- Consistent camera white balance
- Controlled lighting conditions
- Similar water clarity across videos
- HSV augmentation may have added unrealistic color variations

### Our Dataset: "HSV Critical"

**Observation**: Disabling HSV caused -34.3% drop

**Hypothesis**: Our dataset may have:
- Variable camera settings across videos
- Different depths (different color casts)
- Varying water clarity and lighting
- HSV augmentation helps generalize across these conditions

**Evidence**:
- 3 different videos (video_0, video_1, video_2)
- Potentially different cameras or settings
- Underwater imagery inherently variable

**Further investigation**:
- Analyze color distributions across the 3 videos
- Check camera metadata if available
- Compare our dataset characteristics with 1st place dataset

---

## Revised Understanding: Why Box Weight Matters

### YOLOv5 1st Place: box=0.2 (Low)

**Context**: YOLOv5 anchor-based detection struggled with precise small-object localization. Reducing box weight may have helped the model focus more on classification confidence.

### YOLOv11 Our Finding: box=7.5 (Default is Best)

**Context**: YOLOv11 anchor-free detection + decoupled heads already handle small objects well. The default box weight (7.5) is pre-optimized for modern datasets.

**Observation**: ALL box weight reductions hurt performance (-7% to -31%)

**Hypothesis**: YOLOv11n (nano model) with limited capacity benefits from:
- Strong localization signal (high box weight)
- Precise bounding boxes for small objects
- Anchor-free detection making good use of box loss

**Implication**: Don't blindly copy YOLOv5 hyperparameters. YOLOv11 is fundamentally different.

---

## Action Items

### Immediate (This Week)

1. ✅ **Keep baseline hyperparameters** (ablation study validated this)
2. 🔄 **Test 1280px resolution** (biggest expected win)
3. 🔄 **Train baseline @ 30 epochs** (convergence)
4. 🔄 **Implement temporal smoothing** (high ROI, low cost)

### Next Week

5. ⏳ **Optimize confidence threshold for F2**
6. ⏳ **Test YOLOv11s if baseline insufficient**
7. ⏳ **Apply SAHI only if still recall-limited**

### Don't Do

- ❌ Aggressive hyperparameter tuning (defaults are good)
- ❌ Fancy augmentation experiments (ablation study showed this)
- ❌ Heavy ensembling (YOLOv11 reduces need)

---

## Updated Success Metrics

### Realistic Targets (Based on YOLOv11 Capabilities)

**With current approach (640px, YOLOv11n, 30 epochs):**
- Expected mAP50: 0.20-0.25
- Expected Recall: 0.15-0.20
- Expected F2: 0.30-0.40

**With resolution increase (1280px, YOLOv11n, 30 epochs):**
- Expected mAP50: 0.25-0.35 (+25-75%)
- Expected Recall: 0.25-0.35 (+67-133%)
- Expected F2: 0.45-0.60

**With temporal smoothing:**
- Expected F2: +0.05-0.10 boost
- Target F2: 0.50-0.70 ✅

**With confidence threshold tuning:**
- Expected F2: +0.05-0.10 boost
- Final F2: 0.55-0.75 ✅ TARGET REACHED

---

## Key Lessons

1. **Architecture matters more than hyperparameters**
   - YOLOv11 improvements > YOLOv5 tuning tricks

2. **Trust modern defaults**
   - Ultralytics has optimized YOLOv11 on diverse datasets
   - Our ablation study validates this

3. **Context-specific findings still matter**
   - HSV augmentation critical for our specific dataset
   - May differ from 1st place YOLOv5 solution

4. **Focus on high-ROI improvements**
   - Resolution > hyperparameter tuning
   - Temporal smoothing > fancy augmentations
   - Model size > ensemble complexity

5. **Don't blindly copy old solutions**
   - YOLOv5 (2022) ≠ YOLOv11 (2024)
   - Validate findings on your architecture

---

## References

- 1st place Kaggle solution (YOLOv5): https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/qns-trust-cv-1st-place-solution
- Ultralytics YOLOv11 documentation
- YOLOv8-v11 architecture improvements
- SAHI slicing methodology
- Our ablation study results (2025-12-22)

---

**Last Updated**: 2025-12-22
**Status**: Ready to prioritize resolution + temporal smoothing
**Next Action**: Test 1280px resolution training
