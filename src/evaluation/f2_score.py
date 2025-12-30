"""
F2 Score Evaluation for Object Detection.

Using the winning team's algorithm from:
https://www.kaggle.com/haqishen/f2-evaluation/script

F2 Score emphasizes recall over precision (beta=2).
"""

import numpy as np
import pandas as pd


def calculate_iou(box1: dict, box2: dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.

    Args:
        box1: Dict with keys 'x', 'y', 'width', 'height' (COCO format)
        box2: Dict with keys 'x', 'y', 'width', 'height' (COCO format)

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
    y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_f2_single_image(
    pred_boxes: list[dict],
    gt_boxes: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """
    Calculate precision, recall, and F2 for a single image.

    Args:
        pred_boxes: List of predicted boxes [{'x', 'y', 'width', 'height', 'confidence'}, ...]
        gt_boxes: List of ground truth boxes [{'x', 'y', 'width', 'height'}, ...]
        iou_threshold: IoU threshold for matching

    Returns:
        Tuple of (precision, recall, f2_score)
    """
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return None, None, None  # Skip frames with no GT (standard practice)

    if len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0  # No predictions

    if len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0  # False positives only

    # Match predictions to ground truth
    matched_gt = set()
    true_positives = 0

    # Sort predictions by confidence (highest first)
    pred_boxes_sorted = sorted(
        pred_boxes, key=lambda x: x.get("confidence", 1.0), reverse=True
    )

    for pred_box in pred_boxes_sorted:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            true_positives += 1

    precision = true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0
    recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0

    # F2 score (beta=2, emphasizes recall)
    beta = 2
    if precision + recall == 0:
        f2_score = 0.0
    else:
        f2_score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

    return precision, recall, f2_score


def calculate_f2_dataset(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Calculate F2 score across entire dataset.

    Args:
        predictions: DataFrame with columns ['image_id', 'boxes'] where boxes is list of dicts
        ground_truth: DataFrame with columns ['image_id', 'boxes'] where boxes is list of dicts
        iou_threshold: IoU threshold for matching

    Returns:
        Dict with 'precision', 'recall', 'f2' averaged across all images
    """
    # Merge predictions with ground truth
    merged = predictions.merge(
        ground_truth,
        on="image_id",
        how="outer",
        suffixes=("_pred", "_gt"),
    )

    # Debug: Verify merge correctness
    print(f"Merged rows: {len(merged)}")
    print(f"Prediction-only rows: {merged['boxes_gt'].isna().sum()}")
    print(f"GT-only rows: {merged['boxes_pred'].isna().sum()}")

    # Fill NaN with empty lists
    merged["boxes_pred"] = merged["boxes_pred"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    merged["boxes_gt"] = merged["boxes_gt"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Calculate F2 for each image
    precisions = []
    recalls = []
    f2_scores = []

    # Track total counts for debugging
    total_pred_boxes = 0
    total_gt_boxes = 0
    total_tp = 0

    # Track frame type distribution
    frames_both = 0
    frames_pred_only = 0
    frames_gt_only = 0
    frames_empty = 0

    for _, row in merged.iterrows():
        p, r, f2 = calculate_f2_single_image(
            pred_boxes=row["boxes_pred"],
            gt_boxes=row["boxes_gt"],
            iou_threshold=iou_threshold,
        )
        precisions.append(p)
        recalls.append(r)
        f2_scores.append(f2)

        # Count boxes
        n_pred = len(row["boxes_pred"])
        n_gt = len(row["boxes_gt"])
        total_pred_boxes += n_pred
        total_gt_boxes += n_gt

        # Track frame types
        if n_pred > 0 and n_gt > 0:
            frames_both += 1
        elif n_pred > 0:
            frames_pred_only += 1
        elif n_gt > 0:
            frames_gt_only += 1
        else:
            frames_empty += 1

        # Estimate TP (precision * predictions)
        if n_pred > 0 and p is not None:
            total_tp += int(p * n_pred)

    # Filter out None values (frames with no GT)
    valid_scores = [
        (p, r, f) for p, r, f in zip(precisions, recalls, f2_scores) if f is not None
    ]

    if valid_scores:
        valid_precisions, valid_recalls, valid_f2_scores = zip(*valid_scores)
    else:
        valid_precisions = valid_recalls = valid_f2_scores = [0.0]

    # Print debug info
    print("\nFrame Distribution:")
    print(f"  Both pred & GT: {frames_both}")
    print(f"  Pred only: {frames_pred_only}")
    print(f"  GT only: {frames_gt_only}")
    print(f"  Empty: {frames_empty}")
    print("\nBox Statistics:")
    print(f"  Total predictions: {total_pred_boxes}")
    print(f"  Total ground truth: {total_gt_boxes}")
    print(f"  Estimated TP: {total_tp}")
    print("\nEvaluation:")
    print(
        f"  Frames evaluated: {len(valid_scores)} (excluded {len(precisions) - len(valid_scores)} empty frames)"
    )
    print(
        f"  Overall precision: {total_tp / total_pred_boxes if total_pred_boxes > 0 else 0:.4f}"
    )
    print(
        f"  Overall recall: {total_tp / total_gt_boxes if total_gt_boxes > 0 else 0:.4f}"
    )

    return {
        "precision": np.mean(valid_precisions),
        "recall": np.mean(valid_recalls),
        "f2": np.mean(valid_f2_scores),
        "n_images": len(merged),
        "n_evaluated": len(valid_scores),
        "total_predictions": total_pred_boxes,
        "total_ground_truth": total_gt_boxes,
    }
