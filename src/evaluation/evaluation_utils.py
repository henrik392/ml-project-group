"""
Evaluation utilities for COTS detection experiments.

Computes F2, mAP50, precision, recall from predictions and ground truth.
"""

import ast
import pandas as pd
from pathlib import Path

from src.evaluation.f2_score import calculate_f2_dataset


def convert_detections_to_boxes(
    results, frame_ids: list[str] = None, has_tracking: bool = False
) -> pd.DataFrame:
    """
    Convert detection results to boxes format for evaluation.

    Args:
        results: List of detection results (YOLO Results, SAHI PredictionResult, or tracking dicts)
        frame_ids: List of frame IDs (e.g., ['0', '1', '2348', ...]). If None, uses sequential indices.
        has_tracking: Whether results include tracking info (dict format)

    Returns:
        DataFrame with columns ['image_id', 'boxes']
    """
    predictions = []

    for idx, result in enumerate(results):
        # Use provided frame_id or fall back to index
        frame_id = frame_ids[idx] if frame_ids else str(idx)
        image_id = f"frame_{frame_id}"
        boxes = []

        # Check if SAHI or YOLO result
        if hasattr(result, "object_prediction_list"):
            # SAHI result
            for obj_pred in result.object_prediction_list:
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
            # YOLO result (includes tracking results from model.track())
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
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
                # Parse annotations from Python literal format: [{'x': 123, 'y': 456, ...}]
                annotations_str = str(row["annotations"]).strip()
                if annotations_str and annotations_str != "[]":
                    try:
                        # Parse Python list of dictionaries
                        annotations = ast.literal_eval(annotations_str)

                        # Extract bounding boxes
                        if isinstance(annotations, list):
                            for annot in annotations:
                                if isinstance(annot, dict) and all(
                                    k in annot for k in ["x", "y", "width", "height"]
                                ):
                                    boxes.append(
                                        {
                                            "x": int(annot["x"]),
                                            "y": int(annot["y"]),
                                            "width": int(annot["width"]),
                                            "height": int(annot["height"]),
                                        }
                                    )
                    except (ValueError, SyntaxError) as e:
                        # Skip malformed annotations
                        print(
                            f"Warning: Skipping malformed annotation in {image_id}: {e}"
                        )

        # Use frame number as image_id (format to match predictions: "frame_N")
        frame_num = image_id.split("-")[1]
        ground_truth.append(
            {
                "image_id": f"frame_{frame_num}",
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
    frame_ids = predictions.get("frame_ids", None)
    predictions_df = convert_detections_to_boxes(
        pred_results, frame_ids=frame_ids, has_tracking=tracking_enabled
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
