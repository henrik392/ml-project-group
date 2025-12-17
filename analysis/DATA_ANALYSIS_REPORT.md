# TensorFlow Great Barrier Reef - Data Analysis Report

## Competition Overview

**Competition:** TensorFlow - Help Protect the Great Barrier Reef
**Task:** Detect Crown-of-Thorns Starfish (COTS) in underwater video sequences
**Problem Type:** Video Object Detection with Bounding Box Annotations
**Evaluation:** F2 Score (emphasizes recall over precision)

## Executive Summary

This dataset contains underwater video footage for detecting Crown-of-Thorns Starfish (COTS), a coral-eating predator that threatens the Great Barrier Reef. The task is to identify and localize starfish across sequential video frames.

### Key Characteristics

- **23,501 training frames** from 3 videos
- **11,898 starfish annotations** (bounding boxes)
- **79% negative frames** (no starfish present)
- **Temporal structure:** Sequential video frames enabling tracking
- **Image resolution:** 1280x720 pixels (RGB JPEG)

---

## 1. Dataset Structure

### 1.1 File Organization

```
data/
├── train.csv                    # Training annotations
├── test.csv                     # Test metadata (3 frames)
├── train_images/                # Training images
│   ├── video_0/                 # 6,708 images
│   ├── video_1/                 # 8,232 images
│   └── video_2/                 # 8,561 images
├── example_test.npy
├── example_sample_submission.csv
└── greatbarrierreef/            # Competition module
```

### 1.2 CSV Schema

#### Train CSV (23,501 rows × 6 columns)

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | int64 | Video identifier (0, 1, or 2) |
| `sequence` | int64 | Sequence identifier within video |
| `video_frame` | int64 | Frame number within video |
| `sequence_frame` | int64 | Frame number within sequence |
| `image_id` | string | Unique frame identifier (format: `{video_id}-{frame}`) |
| `annotations` | string | JSON list of bounding boxes |

#### Annotation Format

```python
[
    {'x': 559, 'y': 213, 'width': 50, 'height': 32},
    {'x': 598, 'y': 204, 'width': 58, 'height': 32}
]
```

- `x, y`: Top-left corner coordinates
- `width, height`: Bounding box dimensions
- Empty list `[]` indicates no starfish present

---

## 2. Video Distribution

### 2.1 Video Statistics

| Video ID | Frames | Sequences | Frames with Starfish | Starfish % | Total Objects |
|----------|--------|-----------|----------------------|------------|---------------|
| 0 | 6,708 | 7 | 2,143 | 31.9% | 3,065 |
| 1 | 8,232 | 7 | 2,099 | 25.5% | 6,384 |
| 2 | 8,561 | 6 | 677 | 7.9% | 2,449 |
| **Total** | **23,501** | **20** | **4,919** | **20.9%** | **11,898** |

### 2.2 Key Observations

1. **Unequal video lengths:** Video 2 is 28% longer than Video 0
2. **Variable starfish density:** Video 0 has 4× higher starfish presence than Video 2
3. **Sequence structure:** 20 distinct sequences across 3 videos
4. **Frames per sequence:**
   - Mean: 1,175 frames
   - Range: 71 - 2,988 frames
   - Median: 888 frames

---

## 3. Class Distribution & Imbalance

### 3.1 Frame-Level Distribution

```
Positive Frames (with starfish):  4,919 (20.93%)
Negative Frames (no starfish):   18,582 (79.07%)
Imbalance Ratio:                  1:3.78
```

### 3.2 Object Count per Frame

| Objects | Frames | Percentage | Cumulative % |
|---------|--------|------------|--------------|
| 0 | 18,582 | 79.07% | 79.07% |
| 1 | 2,801 | 11.92% | 90.99% |
| 2 | 942 | 4.01% | 95.00% |
| 3 | 374 | 1.59% | 96.59% |
| 4 | 240 | 1.02% | 97.61% |
| 5 | 134 | 0.57% | 98.18% |
| 6-10 | 238 | 1.01% | 99.19% |
| 11-15 | 162 | 0.69% | 99.88% |
| 16-18 | 28 | 0.12% | 100.00% |

**Key Insights:**
- **91% of frames** have 0 or 1 starfish
- **Multi-object frames** (2+ starfish) represent only 9% of data
- Maximum 18 starfish in a single frame (rare occurrence)
- Average **2.42 starfish per positive frame**

---

## 4. Bounding Box Analysis

### 4.1 Size Statistics (11,898 boxes)

| Metric | Width (px) | Height (px) | Area (px²) | Aspect Ratio |
|--------|-----------|-------------|-----------|--------------|
| **Mean** | 47.89 | 42.72 | 2,259.66 | 1.15 |
| **Median** | 45.00 | 40.00 | 1,786.00 | 1.11 |
| **Min** | 17 | 13 | 288 | 0.25 |
| **Max** | 243 | 222 | 52,170 | 4.62 |

### 4.2 Object Scale Distribution

Given image resolution of **1280×720**:

- **Small objects:** Min area = 288 px² (0.03% of image)
- **Medium objects:** Median area = 1,786 px² (0.19% of image)
- **Large objects:** Max area = 52,170 px² (5.67% of image)

**Scale range:** ~181× difference between smallest and largest objects

### 4.3 Shape Characteristics

- **Typical aspect ratio:** ~1.1 (slightly wider than tall)
- **Shape variation:** 0.25 to 4.62 aspect ratio range
- **Mostly square-ish:** Median width (45px) ≈ median height (40px)
- **Outliers exist:** Some very elongated boxes (4.62:1 ratio)

### 4.4 Relative to Image Dimensions

- **Bounding box width:** 1.3% - 19.0% of image width (1280px)
- **Bounding box height:** 1.8% - 30.8% of image height (720px)
- **Typical size:** ~3.5% × 5.6% of image dimensions

---

## 5. Temporal Patterns

### 5.1 Object Sequence Continuity

| Video | Object Sequences | Avg Length (frames) | Max Length (frames) |
|-------|------------------|---------------------|---------------------|
| 0 | 29 | 73.9 | 242 |
| 1 | 23 | 91.3 | 345 |
| 2 | 11 | 61.5 | 290 |

**Observations:**
- Starfish appear in **continuous sequences** averaging 60-90 frames
- Longest sequence: **345 consecutive frames** (Video 1)
- Objects persist across multiple frames, enabling **temporal tracking**

### 5.2 Implications for Modeling

1. **Temporal context is valuable:**
   - Objects appear in long sequences (not random isolated frames)
   - Frame-to-frame motion is likely smooth and predictable
   - Previous/next frames provide strong signal

2. **Video-based approaches recommended:**
   - Temporal models (e.g., 3D CNNs, optical flow)
   - Object tracking algorithms
   - Sequence-to-sequence prediction

3. **Data augmentation opportunities:**
   - Temporal consistency must be maintained
   - Can use neighboring frames for semi-supervised learning

---

## 6. Image Properties

### 6.1 Technical Specifications

- **Resolution:** 1280 × 720 pixels (HD)
- **Format:** JPEG (lossy compression)
- **Color:** RGB (3 channels)
- **Aspect ratio:** 16:9 (widescreen)

### 6.2 Content Characteristics

- **Underwater footage:** Likely blue/green tinted
- **Natural environment:** Variable lighting, water clarity
- **Moving camera:** Video sequences suggest handheld/ROV footage
- **Sequential frames:** Temporal consistency across frames

---

## 7. Data Quality Considerations

### 7.1 Strengths

✅ **Large dataset:** 23,501 labeled frames
✅ **Temporal structure:** Sequential video enables advanced modeling
✅ **High resolution:** 1280×720 provides good detail
✅ **Consistent annotations:** Structured bounding box format
✅ **Real-world data:** Authentic underwater conditions

### 7.2 Challenges

⚠️ **Class imbalance:** 79% negative frames
⚠️ **Multi-scale objects:** 181× size variation
⚠️ **Uneven distribution:** Video 2 has 4× fewer starfish than Video 0
⚠️ **Small objects:** Smallest boxes are only 288 px² (17×13 px)
⚠️ **Crowded scenes:** Up to 18 objects in single frame
⚠️ **Underwater conditions:** Potential issues with:
  - Variable lighting
  - Water turbidity
  - Color distortion
  - Motion blur

---

## 8. Modeling Recommendations

### 8.1 Addressing Class Imbalance

1. **Focal Loss:** Penalize easy negatives, focus on hard examples
2. **Weighted sampling:** Oversample positive frames
3. **Hard negative mining:** Focus on challenging background patches
4. **Synthetic data augmentation:** Create more positive examples

### 8.2 Multi-Scale Detection

1. **Feature Pyramid Networks (FPN):** Handle 181× scale variation
2. **Multi-scale anchors:** Small, medium, large anchor boxes
3. **High-resolution input:** Preserve detail for small objects
4. **Progressive training:** Train on larger inputs progressively

### 8.3 Leveraging Temporal Information

1. **Temporal models:**
   - 3D CNNs (e.g., I3D, SlowFast)
   - Recurrent networks (LSTM, GRU)
   - Temporal Feature Aggregation

2. **Object tracking:**
   - Track-by-detection frameworks
   - Optical flow integration
   - Temporal NMS (non-maximum suppression)

3. **Sequence modeling:**
   - Use neighboring frames as context
   - Temporal consistency losses
   - Motion-aware augmentation

### 8.4 Cross-Validation Strategy

Given **only 3 videos:**
- **Leave-one-video-out CV:** Train on 2 videos, validate on 1
- **Ensures generalization** across different underwater conditions
- **Prevents overfitting** to specific video characteristics

**Critical:** Do NOT use random frame splitting - must preserve video integrity

---

## 9. Winning Solution Insights

Based on the 1st place solution report, key strategies likely included:

### 9.1 Expected Approaches

1. **Ensemble models:** Multiple detectors for robustness
2. **Test-Time Augmentation (TTA):** Multi-scale, flipping
3. **Pseudo-labeling:** Using confident predictions on test set
4. **Post-processing:** Temporal smoothing, NMS tuning
5. **Strong backbones:** EfficientDet, YOLO, or similar

### 9.2 Cross-Validation Trust

- "Trust CV" in title suggests **strong local validation**
- Leave-one-video-out strategy likely critical
- Monitoring F2 score (recall-focused metric)

---

## 10. Data Exploration Recommendations

### 10.1 Visual Analysis Needed

1. **Sample frame visualization:**
   - Plot frames with/without starfish
   - Overlay bounding boxes
   - Analyze edge cases (very small, occluded objects)

2. **Sequence analysis:**
   - Visualize temporal sequences
   - Track individual starfish across frames
   - Identify motion patterns

3. **Error analysis:**
   - Common false positive patterns
   - Missed detections (false negatives)
   - Boundary cases

### 10.2 Statistical Deep Dives

1. **Annotation quality:**
   - Box aspect ratio outliers
   - Unusually large/small boxes
   - Multi-annotator agreement (if available)

2. **Spatial distribution:**
   - Where in frame do starfish typically appear?
   - Edge vs. center distribution
   - Clustering patterns

3. **Inter-video differences:**
   - Color distribution per video
   - Lighting conditions
   - Water clarity metrics

---

## 11. Summary Statistics Table

| Metric | Value |
|--------|-------|
| **Training Frames** | 23,501 |
| **Test Frames** | 13,158 (estimated) |
| **Videos (Train)** | 3 |
| **Sequences** | 20 |
| **Total Annotations** | 11,898 |
| **Positive Frames** | 4,919 (20.93%) |
| **Negative Frames** | 18,582 (79.07%) |
| **Avg Objects/Frame** | 0.506 |
| **Avg Objects/Positive Frame** | 2.419 |
| **Image Resolution** | 1280 × 720 |
| **Bbox Width (median)** | 45 px |
| **Bbox Height (median)** | 40 px |
| **Bbox Area (median)** | 1,786 px² |
| **Scale Variation** | 181× (min to max area) |

---

## 12. Next Steps

### 12.1 Immediate Actions

1. ✅ **Visualize sample frames** with annotations
2. ✅ **Create training/validation split** (leave-one-video-out)
3. ✅ **Establish baseline model** (e.g., YOLOv5, Faster R-CNN)
4. ✅ **Implement evaluation pipeline** (F2 score calculation)
5. ✅ **Analyze prediction errors** on validation set

### 12.2 Advanced Explorations

1. **Temporal modeling experiments:**
   - Compare single-frame vs. multi-frame models
   - Optical flow integration
   - Tracking-based detection

2. **Data augmentation strategy:**
   - Underwater-specific augmentations
   - Temporal consistency preservation
   - Synthetic data generation

3. **Architecture search:**
   - Compare detection frameworks
   - Optimize for small objects
   - Balance speed vs. accuracy

---

## Conclusion

The TensorFlow Great Barrier Reef dataset presents a **challenging video object detection task** with:

- **Severe class imbalance** (4:1 negative to positive ratio)
- **Extreme scale variation** (181× range in object sizes)
- **Temporal structure** enabling advanced video-based approaches
- **Limited videos** (only 3) requiring careful cross-validation

Success will require:
1. **Robust multi-scale detection** to handle small objects
2. **Temporal modeling** to leverage sequential information
3. **Careful validation strategy** (leave-one-video-out)
4. **Class imbalance mitigation** through focal loss/sampling

The winning solution's emphasis on "Trust CV" underscores the importance of **reliable local validation** given the limited number of videos.

---

**Report Generated:** 2025-12-17
**Dataset Version:** TensorFlow Great Barrier Reef (Kaggle)
**Analysis Tool:** Python 3.x with pandas, numpy
