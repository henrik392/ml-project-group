"""
Evaluation utilities for COTS detection experiments.

Computes F2, mAP50, precision, recall from predictions and ground truth.
"""

import pandas as pd
from pathlib import Path

from src.evaluation.f2_score import calculate_f2_dataset


def convert_detections_to_boxes(results, has_tracking: bool = False) -> pd.DataFrame:
    """
    Convert detection results to boxes format for evaluation.

    Args:
        results: List of detection results (YOLO Results, SAHI PredictionResult, or tracking dicts)
        has_tracking: Whether results include tracking info (dict format)

    Returns:
        DataFrame with columns ['image_id', 'boxes']
    """
    predictions = []

    for idx, result in enumerate(results):
        image_id = f"frame_{idx}"
        boxes = []

        # Extract detection object
        if has_tracking:
            # Tracking dict: {'frame_idx', 'detections', 'tracks'}
            det_result = result["detections"]
        else:
            det_result = result

        # Check if SAHI or YOLO result
        if hasattr(det_result, "object_prediction_list"):
            # SAHI result
            for obj_pred in det_result.object_prediction_list:
                bbox = obj_pred.bbox
                boxes.append(
                    {
                        "x": int(bbox.minx),
                        "y": int(bbox.miny),
                        "width": int(bbox.maxx - bbox.minx),
                        "height": int(bbox.maxy - bbox.miny),
                        "confidence": obj_pred.score.value,
                    }
                )
        else:
            # YOLO result
            if det_result.boxes is not None and len(det_result.boxes) > 0:
                for box in det_result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    boxes.append(
                        {
                            "x": int(x1),
                            "y": int(y1),
                            "width": int(x2 - x1),
                            "height": int(y2 - y1),
                            "confidence": conf,
                        }
                    )

        predictions.append({"image_id": image_id, "boxes": boxes})

    return pd.DataFrame(predictions)


def load_ground_truth(fold_id: int, eval_video_id: str) -> pd.DataFrame:
    """
    Load ground truth annotations for evaluation.

    Args:
        fold_id: Fold number (0, 1, 2)
        eval_video_id: Video ID to evaluate (e.g., 'video_0')

    Returns:
        DataFrame with columns ['image_id', 'boxes']
    """
    # Load annotations from train.csv
    train_csv = Path("data/train.csv")
    if not train_csv.exists():
        raise FileNotFoundError(f"Ground truth not found: {train_csv}")

    df = pd.read_csv(train_csv)

    # Filter by video_id (extract number from 'video_0' -> 0)
    video_num = int(eval_video_id.split("_")[1])
    df = df[df["video_id"] == video_num]

    # Group by image_id and collect boxes
    ground_truth = []

    for image_id, group in df.groupby("image_id"):
        boxes = []
        for _, row in group.iterrows():
            if pd.notna(row["annotations"]):
                # Parse annotations: "conf x y width height ..."
                annotations = str(row["annotations"]).strip()
                if annotations:
                    # Remove commas and other non-numeric characters except spaces, dots, and minus
                    annotations = annotations.replace(',', ' ')
                    parts = annotations.split()
                    # Each detection: conf x y width height (5 values)
                    for i in range(0, len(parts), 5):
                        if i + 4 < len(parts):
                            try:
                                boxes.append(
                                    {
                                        "x": int(float(parts[i + 1])),
                                        "y": int(float(parts[i + 2])),
                                        "width": int(float(parts[i + 3])),
                                        "height": int(float(parts[i + 4])),
                                    }
                                )
                            except (ValueError, IndexError):
                                # Skip malformed annotations
                                print(f"Warning: Skipping malformed annotation in {image_id}: {parts[i:i+5]}")

        # Use frame number as image_id
        frame = image_id.split("-")[1]
        ground_truth.append(
            {
                "image_id": frame,
                "boxes": boxes,
            }
        )

    return pd.DataFrame(ground_truth)


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Evaluate predictions against ground truth.

    Args:
        predictions_df: DataFrame with columns ['image_id', 'boxes']
        ground_truth_df: DataFrame with columns ['image_id', 'boxes']
        iou_threshold: IoU threshold for matching (default: 0.5)

    Returns:
        Dict with metrics: f2, map50, recall, precision
    """
    # Calculate F2 score
    f2_results = calculate_f2_dataset(
        predictions=predictions_df,
        ground_truth=ground_truth_df,
        iou_threshold=iou_threshold,
    )

    # For now, use F2 metrics
    # TODO: Add mAP calculation if needed
    metrics = {
        "f2": f2_results["f2"],
        "map50": f2_results["f2"],  # Placeholder
        "recall": f2_results["recall"],
        "precision": f2_results["precision"],
    }

    return metrics


def evaluate_from_config(
    predictions: dict,
    config: dict,
) -> dict:
    """
    Evaluate predictions based on experiment config.

    Args:
        predictions: Dict with 'predictions' key containing results
        config: Experiment config

    Returns:
        Dict with metrics: f2, map50, recall, precision, ms_per_frame
    """
    fold_id = config.get("fold_id", 0)
    eval_video_id = config.get("eval_video_id", f"video_{fold_id}")
    tracking_enabled = config.get("tracking", {}).get("enabled", False)

    print(f"\n{'=' * 80}")
    print(f"Evaluating on {eval_video_id}")
    print(f"{'=' * 80}\n")

    # Convert predictions to DataFrame (works for all modes)
    pred_results = predictions["predictions"]
    predictions_df = convert_detections_to_boxes(
        pred_results, has_tracking=tracking_enabled
    )

    # Load ground truth
    ground_truth_df = load_ground_truth(fold_id, eval_video_id)

    # Evaluate
    metrics = evaluate_predictions(predictions_df, ground_truth_df)

    # Add latency
    metrics["ms_per_frame"] = predictions.get("ms_per_frame", 0)

    print(f"F2: {metrics['f2']:.4f}")
    print(f"mAP50: {metrics['map50']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Latency: {metrics['ms_per_frame']:.2f} ms/frame")

    return metrics
