# Using LLMs for Additional Label Data - Feasibility Analysis

## Question: Could we use LLMs to create more labeled data?

**Short Answer: Theoretically yes, but practically there are better alternatives.**

---

## What You Could Use

### Vision-Language Models (VLMs) for Annotation

1. **GPT-4V / GPT-4o** (OpenAI)
   - Can view images and describe objects
   - Can provide bounding box coordinates
   - API access available

2. **Claude 3.5 Sonnet** (Anthropic)
   - Vision capabilities
   - Can analyze images
   - Potential for object detection

3. **Gemini Pro Vision** (Google)
   - Strong vision understanding
   - Multimodal capabilities

---

## Potential Approaches

### Approach 1: Direct VLM Annotation

**Process:**
```
1. Feed unlabeled frame to VLM
2. Prompt: "Identify all Crown-of-Thorns Starfish and provide bounding boxes"
3. Parse VLM response into structured annotations
4. Add to training data
```

**Pros:**
- Could label the ~11,000 missing frames
- No need for initial model training
- Leverages pre-trained vision understanding

**Cons:**
- ❌ VLMs struggle with precise bounding boxes
- ❌ Expensive (11,000 frames × $0.01-0.05 per image = $110-550)
- ❌ Small underwater objects are challenging for VLMs
- ❌ May hallucinate objects that aren't there
- ❌ Quality likely worse than human annotations
- ❌ Slower than traditional methods

### Approach 2: VLM-Assisted Pseudo-Labeling

**Process:**
```
1. Train initial object detector on existing 23,501 frames
2. Use model to generate candidate boxes on unlabeled frames
3. Use VLM to verify/filter predictions
4. Add high-confidence pseudo-labels to training data
5. Retrain model
```

**Pros:**
- VLM acts as quality filter
- Combines model predictions with VLM reasoning
- Potentially more accurate than pure VLM labeling

**Cons:**
- ❌ Still expensive
- ❌ Complex pipeline
- ❌ VLMs may not be better at verification than confidence thresholds
- ❌ Two-stage process is slow

### Approach 3: Traditional Pseudo-Labeling (Recommended)

**Process:**
```
1. Train strong detector on labeled data
2. Generate predictions on unlabeled frames with high confidence threshold
3. Add pseudo-labels to training data
4. Retrain with combined dataset (labeled + pseudo-labeled)
5. Iterate (teacher-student, self-training)
```

**Pros:**
- ✅ Free (no API costs)
- ✅ Fast inference
- ✅ Model predictions are consistent with task
- ✅ Well-established technique
- ✅ Works well for video data
- ✅ Can use temporal consistency for filtering

**Cons:**
- Requires good initial model
- Can amplify biases if not careful
- Need careful confidence threshold tuning

---

## VLM Limitations for This Task

### 1. Bounding Box Precision

VLMs typically provide:
- **Imprecise coordinates** ("top-left corner around x=500, y=200")
- **Inconsistent formats** (various coordinate systems)
- **No pixel-perfect accuracy** (we need exact x, y, width, height)

Example VLM response:
```
"I can see a Crown-of-Thorns Starfish in the lower right area,
approximately at coordinates (900, 500) with a size of about 50x40 pixels"
```

vs. Required format:
```json
{"x": 890, "y": 513, "width": 47, "height": 42}
```

**Problem:** Off by even 10-20 pixels significantly reduces IoU

### 2. Small Object Detection

Our median starfish: **45×40 pixels** (only 0.19% of image)

VLMs struggle with:
- ❌ Small objects (<1% of image area)
- ❌ Low contrast underwater scenes
- ❌ Multiple small objects in one frame
- ❌ Partially visible/occluded objects

### 3. Domain Specificity

VLMs are trained on general images, not:
- Underwater footage
- Crown-of-Thorns Starfish specifically
- Low-visibility conditions
- Blue/green color casts

**Specialized object detectors** (YOLOv8, EfficientDet) trained on this data will outperform general VLMs.

### 4. Cost Analysis

| Method | Frames | Cost per Frame | Total Cost |
|--------|--------|----------------|------------|
| GPT-4V | 11,000 | $0.01-0.03 | $110-330 |
| Claude Vision | 11,000 | $0.015-0.025 | $165-275 |
| Gemini Pro | 11,000 | $0.005-0.015 | $55-165 |
| **Trained Model** | 11,000 | $0.000 | **$0** |

**With cloud GPU costs:**
- Training detector: ~$5-20 (1-4 hours on cloud GPU)
- Inference on 11k frames: ~$1-5
- **Total: $6-25** vs $55-330 for VLMs

---

## Better Alternatives

### 1. Temporal Propagation (Best for Video Data)

**Leverage the fact that this is video:**

```python
# Use object tracking to propagate labels
for video in videos:
    labeled_frames = get_labeled_frames(video)
    unlabeled_frames = get_unlabeled_frames(video)

    # Use optical flow or tracking algorithm
    propagated_labels = track_objects(
        labeled_frames,
        unlabeled_frames,
        algorithm='DeepSORT'  # or ByteTrack, FairMOT
    )

    # Filter by tracking confidence
    high_confidence = filter_by_confidence(propagated_labels, threshold=0.85)

    # Add to training data
    add_pseudo_labels(high_confidence)
```

**Advantages:**
- ✅ Uses temporal consistency (starfish don't teleport)
- ✅ Free (no API costs)
- ✅ High accuracy for consecutive frames
- ✅ Perfect for this dataset (63-91 frame sequences)

### 2. Semi-Supervised Learning

**Techniques:**
- **MixMatch:** Combines labeled and unlabeled data
- **FixMatch:** Pseudo-labeling with consistency regularization
- **Noisy Student:** Teacher-student with data augmentation

**Framework:**
```python
# Use labeled data (23,501 frames)
labeled_loader = get_labeled_data()

# Use unlabeled frames (11,000 missing frames)
unlabeled_loader = get_unlabeled_data()

# Train with semi-supervised loss
for labeled_batch, unlabeled_batch in zip(labeled_loader, unlabeled_loader):
    # Supervised loss on labeled data
    supervised_loss = compute_loss(model(labeled_batch), labeled_batch.targets)

    # Pseudo-label unlabeled data
    pseudo_labels = model(unlabeled_batch).detach()
    pseudo_labels = filter_by_confidence(pseudo_labels, threshold=0.9)

    # Unsupervised consistency loss
    unsupervised_loss = consistency_loss(
        model(strong_augment(unlabeled_batch)),
        pseudo_labels
    )

    total_loss = supervised_loss + lambda * unsupervised_loss
```

### 3. Data Augmentation (No Labels Needed)

**Generate more training variety without new labels:**

```python
# Underwater-specific augmentations
augmentations = [
    # Color augmentations (simulate water conditions)
    RandomBrightness(0.2),
    RandomContrast(0.2),
    HueSaturationValue(hue_shift=20, sat_shift=30, val_shift=20),

    # Geometric augmentations
    RandomRotate90(),
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15),

    # Blur and noise (simulate water turbidity)
    GaussianBlur(blur_limit=(3, 7)),
    GaussNoise(var_limit=(10, 50)),

    # Advanced augmentations
    MixUp(alpha=0.2),
    CutMix(alpha=1.0),
    Mosaic(p=0.3),  # Combine 4 images
]
```

**Effect:** Effectively 10-50× more training data **without any labeling**

### 4. Active Learning

**Intelligently select which frames to label:**

```python
# Train initial model
model = train_initial_model(labeled_data)

# Score unlabeled frames by uncertainty
uncertainties = []
for frame in unlabeled_frames:
    predictions = model(frame)
    uncertainty = calculate_uncertainty(predictions)  # entropy, variance, etc.
    uncertainties.append((frame, uncertainty))

# Select most uncertain frames (most informative)
top_uncertain = sorted(uncertainties, key=lambda x: x[1], reverse=True)[:1000]

# Option 1: Manual labeling (hire annotators for just these 1,000)
# Option 2: Use VLM only on these 1,000 frames (reduces cost 11×)
# Option 3: Use tracking to propagate from nearest labeled frames
```

**Advantage:** Focus effort on most valuable frames

---

## Experimental Approach: Try VLM Annotation

If you want to experiment with VLM labeling, here's how:

### Step 1: Test VLM on Sample Images

```python
import anthropic
import base64
from pathlib import Path

def test_vlm_annotation(image_path):
    """Test Claude's ability to detect starfish"""

    # Load image
    with open(image_path, 'rb') as f:
        image_data = base64.standard_b64encode(f.read()).decode('utf-8')

    # Prompt engineering
    prompt = """You are an expert marine biologist specializing in Crown-of-Thorns Starfish detection.

Analyze this underwater image and identify ALL Crown-of-Thorns Starfish (COTS) present.

For each starfish, provide:
1. A confidence score (0-1)
2. Bounding box coordinates in the format: x, y, width, height
   - x, y: top-left corner pixel coordinates
   - width, height: box dimensions in pixels
   - Image dimensions: 1280x720 pixels

Return ONLY a JSON array in this exact format:
[
    {"x": 100, "y": 200, "width": 45, "height": 40, "confidence": 0.95},
    {"x": 500, "y": 300, "width": 50, "height": 42, "confidence": 0.87}
]

If no starfish are present, return: []

Be conservative - only annotate objects you're confident are COTS."""

    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    return message.content[0].text

# Test on frames with known annotations
test_images = [
    "data/train_images/video_0/16.jpg",  # Has 1 starfish
    "data/train_images/video_0/35.jpg",  # Has 2 starfish
    "data/train_images/video_0/0.jpg",   # Has 0 starfish
]

for img_path in test_images:
    print(f"\nTesting: {img_path}")
    result = test_vlm_annotation(img_path)
    print(result)
```

### Step 2: Compare VLM vs Ground Truth

```python
import json
import ast

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def evaluate_vlm_annotations(vlm_boxes, ground_truth_boxes, iou_threshold=0.5):
    """Compare VLM predictions to ground truth"""

    # Match predictions to ground truth
    matched = 0
    for gt_box in ground_truth_boxes:
        for vlm_box in vlm_boxes:
            if calculate_iou(gt_box, vlm_box) >= iou_threshold:
                matched += 1
                break

    precision = matched / len(vlm_boxes) if vlm_boxes else 0
    recall = matched / len(ground_truth_boxes) if ground_truth_boxes else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    }

# Run evaluation on sample
train_df = pd.read_csv("data/train.csv")

for idx in [16, 35, 37]:  # Frames with known objects
    row = train_df[train_df['image_id'] == f'0-{idx}'].iloc[0]
    image_path = f"data/train_images/video_0/{idx}.jpg"

    # Get ground truth
    gt_boxes = ast.literal_eval(row['annotations'])

    # Get VLM prediction
    vlm_result = test_vlm_annotation(image_path)
    vlm_boxes = json.loads(vlm_result)

    # Evaluate
    metrics = evaluate_vlm_annotations(vlm_boxes, gt_boxes)

    print(f"\nFrame 0-{idx}:")
    print(f"  Ground truth: {len(gt_boxes)} objects")
    print(f"  VLM detected: {len(vlm_boxes)} objects")
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Recall: {metrics['recall']:.2f}")
    print(f"  F1: {metrics['f1']:.2f}")
```

### Step 3: Decision Criteria

**Use VLM labeling ONLY if:**
- ✅ Precision > 0.85 (few false positives)
- ✅ Recall > 0.80 (finds most objects)
- ✅ Bounding box IoU > 0.60 (reasonably accurate)
- ✅ Cost is justified (better results than alternatives)

**Otherwise:** Use traditional pseudo-labeling or temporal propagation

---

## Recommendation: Hybrid Approach

**Best strategy combines multiple techniques:**

### Phase 1: Strong Baseline (Week 1)
1. Train YOLOv8/EfficientDet on existing 23,501 frames
2. Implement strong data augmentation
3. Leave-one-video-out cross-validation
4. Target F2 > 0.75

### Phase 2: Temporal Propagation (Week 2)
1. Use object tracking (DeepSORT/ByteTrack) to label missing frames
2. Filter by tracking confidence > 0.85
3. Add 3,000-5,000 high-confidence pseudo-labels
4. Retrain model
5. Target F2 > 0.80

### Phase 3: Semi-Supervised Learning (Week 3)
1. Implement FixMatch or Noisy Student
2. Use remaining unlabeled frames
3. Self-training iterations
4. Target F2 > 0.85

### Phase 4: VLM Refinement (Optional)
1. **IF** baseline F2 < 0.80, consider VLM experiment
2. Test on 100 frames first
3. **Only proceed if VLM F1 > 0.80**
4. Use for hard negative mining or uncertainty sampling

---

## Cost-Benefit Analysis

| Approach | Setup Time | Cost | Expected Gain | Recommended |
|----------|-----------|------|---------------|-------------|
| **Data Augmentation** | 1 day | $0 | +5-10% F2 | ✅ Yes |
| **Temporal Propagation** | 2-3 days | $5 | +8-12% F2 | ✅ Yes |
| **Semi-Supervised** | 3-5 days | $10 | +3-8% F2 | ✅ Yes |
| **VLM Labeling** | 5-7 days | $165-330 | +2-5% F2 | ⚠️ Maybe |
| **Active Learning** | 3-4 days | $50-100 | +5-10% F2 | ⚠️ Maybe |

---

## Final Verdict

### Should you use LLMs for labeling? **No, not initially.**

**Better priority order:**
1. ✅ **Strong augmentation** (highest ROI, zero cost)
2. ✅ **Temporal propagation** (perfect for video data)
3. ✅ **Semi-supervised learning** (proven technique)
4. ⚠️ **VLM labeling** (experiment if other methods plateau)

**VLMs are not a silver bullet** for this task. The traditional ML techniques (tracking, pseudo-labeling, augmentation) will give you better results for less cost and complexity.

**When VLMs might make sense:**
- You've exhausted traditional methods
- You have budget for experimentation
- You need to label very specific edge cases
- You're doing active learning and only labeling 500-1000 frames

---

## Bottom Line

**Don't start with VLM labeling.** Build a strong baseline first using:
1. Existing labeled data (23,501 frames)
2. Heavy augmentation
3. Temporal tracking for pseudo-labels

**Only consider VLMs** if you plateau below competitive performance (F2 < 0.80) after trying standard techniques.

**Most likely:** You'll get to F2 > 0.85 without ever needing VLM labeling.
