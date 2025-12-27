"""
Convert Great Barrier Reef annotations to YOLO format.

YOLO format: class_id x_center y_center width height (all normalized 0-1)
GBR format: x, y, width, height (top-left corner, absolute pixels)
"""

import ast
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def convert_to_yolo_format(
    csv_path: str,
    images_dir: str,
    output_dir: str,
    image_width: int = 1280,
    image_height: int = 720,
) -> None:
    """
    Convert GBR annotations to YOLO format.

    Args:
        csv_path: Path to train.csv
        images_dir: Path to train_images/ directory
        output_dir: Output directory for YOLO format data
        image_width: Image width in pixels
        image_height: Image height in pixels
    """
    # Read annotations
    df = pd.read_csv(csv_path)

    # Create output directories
    output_path = Path(output_dir)
    images_out = output_path / "images"
    labels_out = output_path / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(df)} annotations to YOLO format...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row["video_id"]
        video_frame = row["video_frame"]
        image_id = row["image_id"]

        # Parse annotations
        annotations = ast.literal_eval(row["annotations"])

        # Source image path
        src_image = Path(images_dir) / f"video_{video_id}" / f"{video_frame}.jpg"

        if not src_image.exists():
            print(f"Warning: Image not found: {src_image}")
            continue

        # Destination paths
        dst_image = images_out / f"{image_id}.jpg"
        label_file = labels_out / f"{image_id}.txt"

        # Copy image
        if not dst_image.exists():
            shutil.copy(src_image, dst_image)

        # Convert annotations to YOLO format
        yolo_annotations = []
        for ann in annotations:
            # GBR format: x (left), y (top), width, height
            x = ann["x"]
            y = ann["y"]
            w = ann["width"]
            h = ann["height"]

            # Convert to YOLO format (normalized center coordinates)
            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            w_norm = w / image_width
            h_norm = h / image_height

            # Clip to valid range [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            # Class 0 for COTS (Crown-of-Thorns Starfish)
            yolo_annotations.append(
                f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

        # Write label file (empty if no annotations)
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_annotations))

    print("✓ Conversion complete!")
    print(f"  Images: {images_out}")
    print(f"  Labels: {labels_out}")


def main():
    """Main conversion function."""
    # Paths
    csv_path = "data/train.csv"
    images_dir = "data/train_images"
    output_dir = "data/yolo_format"

    # Convert
    convert_to_yolo_format(
        csv_path=csv_path,
        images_dir=images_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
