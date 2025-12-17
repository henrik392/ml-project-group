"""
Annotation Quality Analysis
Checks for potential annotation errors and inconsistencies
"""

import pandas as pd
import numpy as np
import ast
from collections import defaultdict

# Load data
train_df = pd.read_csv("data/train.csv")

# Parse annotations
def parse_annotation(ann_str):
    try:
        return ast.literal_eval(ann_str)
    except:
        return []

train_df['parsed_annotations'] = train_df['annotations'].apply(parse_annotation)
train_df['n_objects'] = train_df['parsed_annotations'].apply(len)

print("=" * 80)
print("ANNOTATION QUALITY ANALYSIS")
print("=" * 80)

# 1. Check for temporal discontinuities (objects suddenly appearing/disappearing)
print("\n1. TEMPORAL DISCONTINUITY ANALYSIS")
print("-" * 80)

for video_id in sorted(train_df['video_id'].unique()):
    video_df = train_df[train_df['video_id'] == video_id].sort_values('video_frame')

    object_counts = video_df['n_objects'].values
    frames = video_df['video_frame'].values

    # Find sudden changes (0 -> many or many -> 0)
    sudden_appearances = []
    sudden_disappearances = []

    for i in range(1, len(object_counts)):
        prev_count = object_counts[i-1]
        curr_count = object_counts[i]

        # Sudden appearance: 0 objects -> 3+ objects
        if prev_count == 0 and curr_count >= 3:
            sudden_appearances.append((frames[i], curr_count))

        # Sudden disappearance: 3+ objects -> 0 objects
        if prev_count >= 3 and curr_count == 0:
            sudden_disappearances.append((frames[i], prev_count))

    print(f"\nVideo {video_id}:")
    print(f"  Sudden appearances (0 -> 3+): {len(sudden_appearances)}")
    if sudden_appearances[:3]:
        print(f"    Examples: {sudden_appearances[:3]}")
    print(f"  Sudden disappearances (3+ -> 0): {len(sudden_disappearances)}")
    if sudden_disappearances[:3]:
        print(f"    Examples: {sudden_disappearances[:3]}")

# 2. Check for bounding box anomalies
print("\n\n2. BOUNDING BOX ANOMALY DETECTION")
print("-" * 80)

all_boxes = []
suspicious_boxes = []
out_of_bounds = []

for idx, row in train_df.iterrows():
    for box in row['parsed_annotations']:
        all_boxes.append(box)

        # Check for out-of-bounds boxes (image is 1280x720)
        if box['x'] < 0 or box['y'] < 0:
            suspicious_boxes.append((row['image_id'], box, "Negative coordinates"))

        if box['x'] + box['width'] > 1280:
            out_of_bounds.append((row['image_id'], box, "Exceeds image width"))

        if box['y'] + box['height'] > 720:
            out_of_bounds.append((row['image_id'], box, "Exceeds image height"))

        # Check for unreasonably small boxes
        if box['width'] < 10 or box['height'] < 10:
            suspicious_boxes.append((row['image_id'], box, "Very small (<10px)"))

        # Check for unreasonably large boxes
        if box['width'] > 300 or box['height'] > 300:
            suspicious_boxes.append((row['image_id'], box, "Very large (>300px)"))

        # Check for unusual aspect ratios
        aspect_ratio = box['width'] / box['height'] if box['height'] > 0 else 0
        if aspect_ratio > 3 or aspect_ratio < 0.33:
            suspicious_boxes.append((row['image_id'], box, f"Unusual aspect ratio: {aspect_ratio:.2f}"))

print(f"\nTotal boxes analyzed: {len(all_boxes)}")
print(f"Out-of-bounds boxes: {len(out_of_bounds)}")
if out_of_bounds[:3]:
    print(f"  Examples:")
    for img_id, box, reason in out_of_bounds[:3]:
        print(f"    {img_id}: {box} - {reason}")

print(f"\nSuspicious boxes: {len(suspicious_boxes)}")
if suspicious_boxes[:5]:
    print(f"  Examples:")
    for img_id, box, reason in suspicious_boxes[:5]:
        print(f"    {img_id}: {box} - {reason}")

# 3. Check for missing frames in sequences
print("\n\n3. SEQUENCE CONTINUITY CHECK")
print("-" * 80)

for video_id in sorted(train_df['video_id'].unique()):
    video_df = train_df[train_df['video_id'] == video_id].sort_values('video_frame')

    # Check if frames are sequential
    frames = video_df['video_frame'].values
    expected_frames = np.arange(frames[0], frames[-1] + 1)

    missing_frames = set(expected_frames) - set(frames)

    print(f"\nVideo {video_id}:")
    print(f"  Frame range: {frames[0]} to {frames[-1]}")
    print(f"  Total frames: {len(frames)}")
    print(f"  Expected frames: {len(expected_frames)}")
    print(f"  Missing frames: {len(missing_frames)}")
    if missing_frames:
        print(f"    First missing: {sorted(missing_frames)[:10]}")

# 4. Check for annotation consistency across sequences
print("\n\n4. TEMPORAL TRACKING CONSISTENCY")
print("-" * 80)

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

# Sample check: look at consecutive frames with objects
large_jumps = []

for video_id in sorted(train_df['video_id'].unique()):
    video_df = train_df[train_df['video_id'] == video_id].sort_values('video_frame')

    jump_count = 0
    for i in range(1, min(len(video_df), 100)):  # Sample first 100 frames
        prev_boxes = video_df.iloc[i-1]['parsed_annotations']
        curr_boxes = video_df.iloc[i]['parsed_annotations']

        # If both frames have exactly 1 object, check if it moved too much
        if len(prev_boxes) == 1 and len(curr_boxes) == 1:
            iou = calculate_iou(prev_boxes[0], curr_boxes[0])

            # If IoU is very low, object jumped significantly
            if iou < 0.1:
                jump_count += 1
                if jump_count <= 3:
                    frame_num = video_df.iloc[i]['video_frame']
                    large_jumps.append((video_id, frame_num, iou))

    print(f"\nVideo {video_id}:")
    print(f"  Large object jumps (IoU < 0.1): {jump_count} in first 100 frames")

# 5. Statistical outliers
print("\n\n5. STATISTICAL OUTLIERS")
print("-" * 80)

if all_boxes:
    widths = [box['width'] for box in all_boxes]
    heights = [box['height'] for box in all_boxes]

    width_mean = np.mean(widths)
    width_std = np.std(widths)
    height_mean = np.mean(heights)
    height_std = np.std(heights)

    # Find boxes > 3 standard deviations from mean
    extreme_outliers = []
    for idx, row in train_df.iterrows():
        for box in row['parsed_annotations']:
            width_zscore = abs(box['width'] - width_mean) / width_std
            height_zscore = abs(box['height'] - height_mean) / height_std

            if width_zscore > 3 or height_zscore > 3:
                extreme_outliers.append((row['image_id'], box, f"Z-score: W={width_zscore:.2f}, H={height_zscore:.2f}"))

    print(f"\nExtreme outliers (>3 std dev): {len(extreme_outliers)}")
    if extreme_outliers[:5]:
        print(f"  Examples:")
        for img_id, box, reason in extreme_outliers[:5]:
            print(f"    {img_id}: {box} - {reason}")

# 6. Check object count distribution within sequences
print("\n\n6. WITHIN-SEQUENCE OBJECT COUNT VARIANCE")
print("-" * 80)

for video_id in sorted(train_df['video_id'].unique()):
    video_df = train_df[train_df['video_id'] == video_id]

    for sequence_id in sorted(video_df['sequence'].unique()):
        seq_df = video_df[video_df['sequence'] == sequence_id].sort_values('sequence_frame')

        if len(seq_df) < 10:  # Skip short sequences
            continue

        object_counts = seq_df['n_objects'].values

        # Check variance
        if np.std(object_counts) > 3:  # High variance suggests potential issues
            print(f"\nVideo {video_id}, Sequence {sequence_id}:")
            print(f"  Frames: {len(seq_df)}")
            print(f"  Object count: mean={np.mean(object_counts):.2f}, std={np.std(object_counts):.2f}")
            print(f"  Range: {np.min(object_counts)} to {np.max(object_counts)}")

            # Show distribution
            unique, counts = np.unique(object_counts, return_counts=True)
            print(f"  Distribution: {dict(zip(unique, counts))}")

print("\n" + "=" * 80)
print("SUMMARY: POTENTIAL ANNOTATION ISSUES")
print("=" * 80)

print(f"""
⚠️  Findings:
   - Out-of-bounds boxes: {len(out_of_bounds)}
   - Suspicious boxes: {len(suspicious_boxes)}
   - Extreme outliers: {len(extreme_outliers)}
   - Large object jumps: {len(large_jumps)}

💡 Recommendations:
   1. Manual inspection of suspicious boxes recommended
   2. Consider these issues during model training
   3. Potential label noise should be accounted for
   4. Cross-validation will help identify systematic annotation errors
""")
