"""
Mock Kaggle Submission - Demonstrates What Would Be Submitted

Since the competition API is broken (Python 3.7 → 3.12 incompatibility),
this script demonstrates the submission format and generates predictions
that WOULD be submitted if the API worked.

Use this for your project report/presentation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def format_predictions_for_submission(results, conf_threshold=0.15):
    """
    Format YOLO predictions into Kaggle submission format.

    Format: "conf x y width height conf x y width height ..."
    """
    if results is None or len(results) == 0:
        return ""

    annotations = []
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return ""

    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        # Get box coordinates (xyxy format)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Convert to x, y, width, height (COCO format)
        x = int(x1)
        y = int(y1)
        width = int(x2 - x1)
        height = int(y2 - y1)

        # Format: "conf x y width height"
        annotations.append(f"{conf:.2f} {x} {y} {width} {height}")

    return " ".join(annotations)


def create_mock_submission(
    model_path: str,
    val_images_dir: str,
    output_csv: str = "mock_submission.csv",
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.3,
):
    """
    Create a mock submission file using validation set.

    This demonstrates what WOULD be submitted if the competition API worked.
    """
    print(f"\n{'='*80}")
    print("MOCK KAGGLE SUBMISSION - DEMONSTRATION")
    print(f"{'='*80}\n")
    print("NOTE: This is a demonstration. The actual competition API is broken.")
    print("      We're using validation images to show submission format.\n")

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Get validation images
    val_images = sorted(Path(val_images_dir).glob("*.jpg"))
    print(f"Found {len(val_images)} validation images\n")

    # Generate predictions
    submissions = []

    for idx, img_path in enumerate(val_images):
        # Run prediction
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            verbose=False,
        )

        # Format predictions
        annotations = format_predictions_for_submission(results, conf_threshold)

        submissions.append({
            "row_num": idx,
            "image_id": img_path.stem,
            "annotations": annotations,
        })

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(val_images)} images")

    # Create submission DataFrame
    submission_df = pd.DataFrame(submissions)

    # Save to CSV
    submission_df.to_csv(output_csv, columns=["row_num", "annotations"], index=False)

    print(f"\n{'='*80}")
    print(f"Mock submission saved to: {output_csv}")
    print(f"{'='*80}\n")

    # Statistics
    total_predictions = submission_df["annotations"].apply(lambda x: len(x.split()) // 5).sum()
    images_with_detections = (submission_df["annotations"] != "").sum()

    print("Submission Statistics:")
    print(f"  Total images: {len(submission_df)}")
    print(f"  Images with detections: {images_with_detections}")
    print(f"  Images without detections: {len(submission_df) - images_with_detections}")
    print(f"  Total COTS detected: {total_predictions}")
    print(f"  Average detections per image: {total_predictions / len(submission_df):.2f}")
    print(f"  Detection rate: {images_with_detections / len(submission_df) * 100:.1f}%\n")

    # Show sample predictions
    print("Sample predictions (first 10):")
    print(submission_df.head(10).to_string(index=False))
    print()

    return submission_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create mock Kaggle submission")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/train/yolo11n_fold02/weights/best.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--val-images",
        type=str,
        default="data/folds/fold_0/val/images",
        help="Path to validation images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mock_submission.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.3,
        help="IoU threshold for NMS",
    )

    args = parser.parse_args()

    create_mock_submission(
        model_path=args.model,
        val_images_dir=args.val_images,
        output_csv=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
