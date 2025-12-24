"""
Generate predictions for Kaggle submission.

Uses trained YOLOv11 model to predict COTS on test images.
Outputs submission.csv in the required format.
"""

import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def format_predictions(results, conf_threshold=0.25):
    """
    Format YOLO predictions for Kaggle submission.

    Args:
        results: YOLO prediction results
        conf_threshold: Minimum confidence threshold

    Returns:
        String in format: "conf x y width height conf x y width height ..."
        Empty string if no detections
    """
    if results is None or len(results) == 0:
        return ""

    annotations = []

    # Get boxes, confidences
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


def predict_test_set(
    model_path: str,
    test_csv: str = "data/test.csv",
    image_dir: str = "data/train_images",
    output_csv: str = "submission.csv",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.3,
    imgsz: int = 640,
) -> None:
    """
    Generate predictions on test set and create submission file.

    Args:
        model_path: Path to trained model weights (.pt file)
        test_csv: Path to test.csv
        image_dir: Directory containing test images
        output_csv: Output submission file path
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for NMS (per winning solution: 0.3)
        imgsz: Image size for inference
    """
    print(f"\n{'='*80}")
    print(f"Generating Kaggle Submission")
    print(f"{'='*80}\n")

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Load test.csv
    print(f"Loading test data: {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"Found {len(test_df)} test images")

    # Prepare submission
    submissions = []

    for idx, row in test_df.iterrows():
        image_id = row["image_id"]
        video_id = row["video_id"]

        # Construct image path
        # Format: data/train_images/video_X/FRAME.jpg
        # image_id format: "X-FRAME"
        frame = image_id.split("-")[1]
        image_path = Path(image_dir) / f"video_{video_id}" / f"{frame}.jpg"

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            annotations = ""
        else:
            # Run prediction
            results = model.predict(
                source=str(image_path),
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                verbose=False,
            )

            # Format predictions
            annotations = format_predictions(results, conf_threshold)

        submissions.append({"row_num": idx, "annotations": annotations})

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} images")

    # Create submission DataFrame
    submission_df = pd.DataFrame(submissions)

    # Save to CSV
    submission_df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Submission saved to: {output_csv}")
    print(f"{'='*80}\n")

    # Show sample predictions
    print("Sample predictions:")
    print(submission_df.head())


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Generate Kaggle submission")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights (.pt file)",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/test.csv",
        help="Path to test.csv",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/train_images",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output submission file path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25, lower for higher recall)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.3,
        help="IoU threshold for NMS (default: 0.3 per winning solution)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference",
    )

    args = parser.parse_args()

    predict_test_set(
        model_path=args.model,
        test_csv=args.test_csv,
        image_dir=args.image_dir,
        output_csv=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
