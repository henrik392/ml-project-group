# COTS Detection: YOLOv5 → YOLOv11 → SAHI → ByteTrack

**Tsinghua University - Machine Learning Course Project**

---

## Abstract

[TODO: 150-word summary of problem, approach, key findings]

Crown-of-Thorns Starfish (COTS) pose a significant threat to coral reef ecosystems. This project investigates modern object detection approaches for COTS identification in underwater video, comparing historical YOLOv5 baselines with contemporary YOLOv11 architectures. We employ leave-one-video-out cross-validation (3 folds) and optimize for the F2 metric to prioritize recall. Our experiments explore: (1) YOLOv5 vs. YOLOv11 baseline performance, (2) minimal YOLOv11 ablations (confidence, IoU, resolution), (3) SAHI tiled inference for accuracy/speed tradeoffs, and (4) ByteTrack temporal context for video enhancement. Results demonstrate [TODO: key findings]. This work establishes a rigorous, reproducible baseline for underwater object detection research.

---

## 1. Introduction

### 1.1 Problem Statement

Crown-of-Thorns Starfish (COTS, *Acanthaster planci*) are a major threat to coral reefs, particularly in the Great Barrier Reef ecosystem. Population outbreaks can rapidly destroy large coral reef areas, necessitating early detection and intervention. Manual monitoring is labor-intensive and limited in scale, motivating the development of automated detection systems.

### 1.2 Dataset

**TensorFlow Great Barrier Reef Dataset** (Kaggle 2022):
- **Format**: Underwater video sequences
- **Size**: 23,501 annotated frames from 3 videos
- **Challenge**: Small objects, underwater lighting variation, class imbalance

### 1.3 Evaluation Metric

**F2 Score** (recall-weighted):
```
F2 = (1 + 2²) × (precision × recall) / (2² × precision + recall)
F2 = 5 × (precision × recall) / (4 × precision + recall)
```

The F2 metric weighs recall 5× more than precision, aligning with the ecological objective of minimizing false negatives (missing starfish is costlier than false alarms).

### 1.4 Evaluation Protocol

**Leave-one-video-out cross-validation (3 folds)**:
- Hold out 1 full video for evaluation
- Train on the other 2 videos
- No frame leakage between train/val splits
- Report results across all 3 folds

This protocol ensures video-aware evaluation, preventing data leakage from temporal correlation in video frames.

---

## 2. Related Work

### 2.1 YOLOv5 Winning Solutions (2022)

The original Kaggle competition was won using YOLOv5 with:
- Modified hyperparameters: `box=0.2`, `iou_t=0.3`
- Augmentations: rotation, mixup, Transpose (NO HSV)
- SAHI for small object detection
- Temporal post-processing

### 2.2 Modern Object Detection

**YOLOv11 (2024)** introduces:
- Anchor-free detection → better small-object localization
- Decoupled heads → cleaner classification vs box regression
- Better multi-scale features → fewer missed tiny objects
- NMS-free training (v10+) → fewer duplicate boxes

These architectural improvements reduce reliance on hyperparameter tuning and heavy ensembling.

---

## 3. Methods

### 3.1 Dataset & Video-Aware Split

**Data preparation**:
1. Convert annotations to YOLO format (class x_center y_center width height)
2. Create 3-fold leave-one-video-out splits:
   - Fold 0: Train on video_1, video_2 → Validate on video_0
   - Fold 1: Train on video_0, video_2 → Validate on video_1
   - Fold 2: Train on video_0, video_1 → Validate on video_2

**Rationale**: Video frames are temporally correlated. Holding out full videos prevents data leakage and provides realistic evaluation.

### 3.2 Models

#### 3.2.1 YOLOv5n (Historical Baseline)

- **Parameters**: ~1.9M
- **Architecture**: Anchor-based detection
- **Training**: Default YOLOv5 hyperparameters
- **Purpose**: Establish 2022 SOTA baseline for comparison

#### 3.2.2 YOLOv11n (Modern Baseline)

- **Parameters**: 2.59M
- **Architecture**: Anchor-free detection, decoupled heads
- **Training**: Default YOLOv11 hyperparameters
- **Augmentation**: HSV, translation, scale, fliplr, mosaic, auto_augment
- **Device**: M4 Max (MPS backend)

### 3.3 SAHI Tiled Inference

**Slicing Aided Hyper Inference (SAHI)**:
- Divides images into overlapping tiles
- Runs inference on each tile
- Merges predictions with NMS
- **Trade-off**: 2-6× slower, but better small object detection

**Configuration**:
- Slice size: TBD
- Overlap ratio: TBD

### 3.4 ByteTrack Temporal Context

**ByteTrack** via `model.track`:
- Lightweight object tracking across video frames
- **Purpose**: Temporal recall enhancement (not ID tracking)
- **Hypothesis**: Boosting detections in nearby frames improves F2 score

**Freeze training rule**: Use same trained weights, vary only inference settings (standard vs. SAHI, with/without ByteTrack).

---

## 4. Experiments & Results

### 4.1 Historical Baseline (YOLOv5 vs YOLOv11)

**Objective**: Compare 2022 SOTA (YOLOv5) with modern architecture (YOLOv11) under identical conditions.

| Model | F2 | mAP50 | Recall | Precision | ms/frame |
|-------|----|----|---|---|---|
| YOLOv5n | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| YOLOv11n | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

**Observation**: [TODO: describe which model performs better and by how much]

**Hypothesis**: [TODO: explain why one architecture outperforms the other]

### 4.2 YOLOv11 Ablation (conf/iou/imgsz)

**Objective**: Identify optimal inference hyperparameters for F2 metric.

#### 4.2.1 Confidence Threshold Sweep

| conf | F2 | Recall | Precision |
|------|----|----|---|
| 0.001 | [TODO] | [TODO] | [TODO] |
| 0.01 | [TODO] | [TODO] | [TODO] |
| 0.05 | [TODO] | [TODO] | [TODO] |
| 0.1 | [TODO] | [TODO] | [TODO] |
| 0.25 | [TODO] | [TODO] | [TODO] |

**Finding**: [TODO: optimal confidence threshold for F2]

#### 4.2.2 IoU Threshold Sweep

| iou | F2 | Recall | Precision |
|-----|----|----|---|
| 0.3 | [TODO] | [TODO] | [TODO] |
| 0.45 | [TODO] | [TODO] | [TODO] |
| 0.7 | [TODO] | [TODO] | [TODO] |

**Finding**: [TODO: optimal IoU threshold]

#### 4.2.3 Image Resolution Sweep

| imgsz | F2 | mAP50 | ms/frame |
|-------|----|----|---|
| 640 | [TODO] | [TODO] | [TODO] |
| 1280 | [TODO] | [TODO] | [TODO] |

**Finding**: [TODO: accuracy vs. speed tradeoff]

### 4.3 Inference Strategy (Standard vs SAHI)

**Objective**: Quantify SAHI's accuracy/speed tradeoff.

| Inference | F2 | mAP50 | ms/frame | Notes |
|-----------|----|----|---|---|
| Standard | [TODO] | [TODO] | [TODO] | Baseline |
| SAHI | [TODO] | [TODO] | [TODO] | +X% F2, -Y% speed |

**Observation**: [TODO: does SAHI improve small object detection?]

**Trade-off**: [TODO: is the speed penalty justified by accuracy gain?]

### 4.4 Temporal Context (ByteTrack)

**Objective**: Evaluate temporal tracking for video F2 enhancement.

| Method | F2 | Recall | Precision |
|--------|----|----|---|
| Standard inference | [TODO] | [TODO] | [TODO] |
| + ByteTrack | [TODO] | [TODO] | [TODO] |
| SAHI | [TODO] | [TODO] | [TODO] |
| SAHI + ByteTrack | [TODO] | [TODO] | [TODO] |

**Observation**: [TODO: does temporal context improve F2?]

**Hypothesis**: [TODO: explain why temporal tracking helps/doesn't help]

---

## 5. Qualitative Analysis

### 5.1 Success Cases

[TODO: 2×2 grid showing successful detections across methods]
- YOLOv5 | YOLOv11 | SAHI | SAHI+ByteTrack

### 5.2 Failure Cases

**Categories of failure**:
1. **Small objects**: Starfish <10 pixels
2. **Occlusion**: Partially hidden starfish
3. **Lighting**: Extreme underwater color shifts
4. **Motion blur**: Fast camera movement

[TODO: Gallery of failure examples with annotations]

---

## 6. Conclusion

### 6.1 Key Findings

1. [TODO: YOLOv5 vs YOLOv11 comparison]
2. [TODO: Optimal confidence/IoU thresholds for F2]
3. [TODO: SAHI accuracy/speed tradeoff]
4. [TODO: ByteTrack temporal enhancement]

### 6.2 Limitations

1. **Dataset size**: Only 3 videos limits generalization
2. **Single model size**: Only tested YOLOv11n (nano)
3. **No ensemble**: Winning solutions used multi-model ensembles
4. **Local evaluation**: No Kaggle leaderboard verification

### 6.3 Future Work

1. **Model scaling**: Test YOLOv11-s/m/l for accuracy improvements
2. **Ensemble methods**: Combine multiple model predictions
3. **Advanced temporal**: Test more sophisticated tracking algorithms
4. **Deployment**: Real-time inference optimization

---

## References

1. Ultralytics YOLOv11 Documentation: https://docs.ultralytics.com/models/yolo11/
2. Kaggle TensorFlow Great Barrier Reef Competition: https://www.kaggle.com/c/tensorflow-great-barrier-reef
3. 1st Place Solution (YOLOv5): https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/discussion/307768
4. SAHI: Slicing Aided Hyper Inference: https://github.com/obss/sahi
5. ByteTrack: Multi-Object Tracking by Associating Every Detection Box

---

## Appendix: Prior Experiments & Lessons Learned

Prior to establishing the current experimental storyline, extensive preliminary work was conducted to understand baseline performance and validate hyperparameter choices.

### A.1 Initial Baseline Training

**Configuration**: YOLOv11n @ 640px, M4 Max (MPS), default hyperparameters

**Results**:
- **Best performance**: Epoch 10 - mAP50 = 0.126, Precision = 0.471, Recall = 0.114
- **Critical finding**: Very low recall (11.4%) - model misses 88.6% of starfish
- **Training issue**: Performance degraded after epoch 10 (likely due to training interruption/resume)

**Observation**: High precision (47%) but critically low recall indicates the model is conservative. The F2 metric (recall-weighted) requires substantial recall improvement.

### A.2 Systematic Ablation Study

**Design**: 8 experiments, 5 epochs each, Fold 0, changing ONE parameter from baseline

| Experiment | Parameter Changed | mAP50 | Change | Conclusion |
|------------|------------------|-------|--------|------------|
| baseline_control | None | 0.1260 | 0.0% | Reference |
| **mixup_0.1** | +mixup | 0.1262 | **+0.1%** | Neutral, safe to use |
| box_5.5 | box 7.5→5.5 | 0.1173 | -6.9% | Hurts |
| box_5.0 | box 7.5→5.0 | 0.1167 | -7.3% | Hurts |
| box_6.0 | box 7.5→6.0 | 0.1072 | -14.9% | Hurts |
| box_6.5 | box 7.5→6.5 | 0.0870 | -30.9% | Severely hurts |
| rotation_10 | +10° rotation | 0.1100 | -12.7% | Hurts (works on MPS, but degrades performance) |
| **no_hsv** | Disable HSV | **0.0828** | **-34.3%** | **CRITICAL - HSV is essential** |

**Key Findings**:

1. **HSV augmentation is critical** (-34.3% when disabled)
   - **Hypothesis**: Underwater imagery exhibits extreme color variation due to depth, lighting, and water clarity. HSV augmentation helps generalize across these conditions.
   - **Implication**: Explains why a prior failed experiment (Phase 2) that disabled HSV saw -47% degradation.

2. **Default box weight (7.5) is optimal**
   - ALL reductions (7.5 → 6.5, 6.0, 5.5, 5.0) hurt performance (-7% to -31%)
   - **Hypothesis**: YOLOv11n (nano model) with limited capacity benefits from strong localization signal. Anchor-free detection makes good use of high box weight.

3. **Rotation augmentation works on MPS but hurts performance** (-12.7%)
   - **Clarification**: Technical compatibility confirmed (no crashes), but modeling performance degrades
   - **Hypothesis**: COTS may have preferred orientations in video frames, or rotation distorts small objects

4. **Mixup augmentation is neutral** (+0.1%)
   - Safe to enable, but minimal impact at 5 epochs

### A.3 YOLOv11 vs. YOLOv5: Architecture Differences

**Critical realization**: The 2022 Kaggle competition used YOLOv5. Winning solutions optimized for YOLOv5 limitations do NOT directly apply to YOLOv11.

| Feature | YOLOv5 (2022) | YOLOv11 (2024) | Impact |
|---------|---------------|----------------|--------|
| Detection | Anchor-based | **Anchor-free** | Better small-object localization |
| Heads | Coupled | **Decoupled** | Cleaner classification/box separation |
| Multi-scale | Basic FPN | **Enhanced** | Fewer missed tiny objects |
| NMS | Post-processing | **NMS-free training** | Fewer duplicate boxes |
| Defaults | Needed tuning | **Pre-optimized** | Strong out-of-box |

**Why 2022 winning solution used `box=0.2` (not applicable to YOLOv11)**:
- YOLOv5 anchor-based detection struggled with precise small-object localization
- Reducing box weight helped focus on classification confidence
- YOLOv11 anchor-free + decoupled heads already handle small objects well
- Our ablation proves: YOLOv11 default `box=7.5` is optimal

**Why 2022 winning solution disabled HSV (opposite of our finding)**:
- **Hypothesis**: Their dataset may have had consistent camera settings, controlled lighting, or similar water clarity
- **Our dataset**: 3 different videos with potentially variable camera settings, depths, and lighting
- HSV augmentation helps generalize across our variable conditions

**Conclusion**: Trust YOLOv11 defaults and focus on **resolution**, **model scaling**, and **temporal context** instead of hyperparameter tuning.

### A.4 Technical Issues Encountered

#### MPS Backend Compatibility

**Three-way augmentation conflict**:
- ✅ rotation + mosaic: Works
- ✅ mixup + mosaic: Works
- ❌ rotation + mixup + mosaic: **TAL RuntimeError** (shape mismatch in Task-Aligned Assigner)

**Hypothesis**: Augmentation pipeline may not properly track tensor shapes when all three apply in sequence. MPS backend may use different tensor layouts than CUDA, making shape mismatches more likely.

**Resolution**: Use rotation OR mixup, not both simultaneously (for MPS training).

#### Memory Management

**Issue**: Long training runs (30+ epochs) on MPS prone to OOM crashes (exit code 137)
**Mitigation**: Use auto-retry with checkpoint resumption, or train on cloud GPUs

### A.5 Evaluation Protocol Clarification

**Issue**: Kaggle API incompatibility (Python 3.7 binary on Python 3.12 environment)
**Solution**: Local leave-one-video-out cross-validation (3 folds)

**Rationale**:
- More rigorous than single test set evaluation
- Prevents overfitting to leaderboard
- Industry-standard evaluation method
- No frame leakage between train/val due to video-aware splitting

### A.6 Lessons Learned

**Methodological**:
1. **Test one change at a time** - ablation studies reveal true parameter effects
2. **Validate architecture-specific findings** - don't blindly copy YOLOv5 solutions for YOLOv11
3. **Trust modern defaults** - YOLOv11 hyperparameters are pre-optimized
4. **Context matters** - HSV critical for our dataset but disabled in 2022 winner

**Technical**:
1. **Recall is the bottleneck** for F2 metric (not precision)
2. **Small object detection** requires resolution scaling (640px → 1280px)
3. **Temporal context** is high-ROI for video data
4. **MPS backend works** but has quirks (three-way augmentation conflict, memory leaks)

**Strategic**:
1. **Focus on high-impact improvements**: Resolution > hyperparameters
2. **Temporal smoothing** > fancy augmentations
3. **Model scaling** > ensemble complexity
4. **Clarity > complexity** - if defaults win, report that and move on

---

**Document Status**: Draft (Sections 4-5 require experimental results)
**Last Updated**: 2025-12-27
**Course**: Machine Learning, Tsinghua University
**Deadline**: 2026-01-07
