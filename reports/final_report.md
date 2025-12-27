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

## Appendix: Prior Ablation Study (HSV, box weight)

### Background

Prior to establishing this experimental storyline, an ablation study was conducted on YOLOv11 hyperparameters and augmentations (8 experiments, 5 epochs each).

### Key Findings

1. **HSV augmentation is critical**: Disabling HSV caused -34.3% mAP50 degradation
2. **Box weight**: Default (7.5) is optimal; reductions to 6.5, 6.0, 5.5, 5.0 all hurt performance (-7% to -31%)
3. **Rotation augmentation**: 10° rotation caused -12.7% performance drop
4. **Mixup augmentation**: Neutral (+0.1% vs baseline)

### Implication

These findings informed the decision to **keep YOLOv11 default hyperparameters** for the main experimental arc. Over-tuning hyperparameters showed diminishing or negative returns.

**Full analysis**: See `reports/ablation_study_analysis.md`

---

**Document Status**: Draft (Sections 4-5 require experimental results)
**Last Updated**: 2025-12-27
**Course**: Machine Learning, Tsinghua University
**Deadline**: 2026-01-07
