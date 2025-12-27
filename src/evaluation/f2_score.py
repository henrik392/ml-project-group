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
        return 1.0, 1.0, 1.0  # Perfect score if both empty

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

    for _, row in merged.iterrows():
        p, r, f2 = calculate_f2_single_image(
            pred_boxes=row["boxes_pred"],
            gt_boxes=row["boxes_gt"],
            iou_threshold=iou_threshold,
        )
        precisions.append(p)
        recalls.append(r)
        f2_scores.append(f2)

    return {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f2": np.mean(f2_scores),
        "n_images": len(merged),
    }


def yolo_to_coco_format(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_width: int = 1280,
    img_height: int = 720,
) -> dict:
    """
    Convert YOLO format (normalized) to COCO format (absolute pixels).

    Args:
        x_center: Normalized x center (0-1)
        y_center: Normalized y center (0-1)
        width: Normalized width (0-1)
        height: Normalized height (0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Dict with keys 'x', 'y', 'width', 'height' (COCO format, top-left corner)
    """
    # Convert to absolute pixels
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height

    # Convert to top-left corner
    x = x_center_abs - width_abs / 2
    y = y_center_abs - height_abs / 2

    return {
        "x": int(x),
        "y": int(y),
        "width": int(width_abs),
        "height": int(height_abs),
    }


if __name__ == "__main__":
    # Example usage
    pred_boxes = [
        {"x": 100, "y": 100, "width": 50, "height": 50, "confidence": 0.9},
        {"x": 200, "y": 200, "width": 60, "height": 60, "confidence": 0.8},
    ]
    gt_boxes = [
        {"x": 105, "y": 105, "width": 48, "height": 48},
        {"x": 205, "y": 205, "width": 58, "height": 58},
    ]

    p, r, f2 = calculate_f2_single_image(pred_boxes, gt_boxes)
    print(f"Precision: {p:.3f}")
    print(f"Recall: {r:.3f}")
    print(f"F2 Score: {f2:.3f}")
