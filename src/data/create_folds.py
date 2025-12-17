"""
Create 3-fold cross-validation splits by video_id (leave-one-video-out).

Fold 0: Train on video_1, video_2 → Validate on video_0
Fold 1: Train on video_0, video_2 → Validate on video_1
Fold 2: Train on video_0, video_1 → Validate on video_2
"""

import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def create_fold_splits(
    yolo_data_dir: str = "data/yolo_format",
    output_dir: str = "data/folds",
) -> None:
    """
    Create 3-fold splits by video_id.

    Args:
        yolo_data_dir: Directory with YOLO format data (images/ and labels/)
        output_dir: Output directory for fold splits
    """
    yolo_path = Path(yolo_data_dir)
    output_path = Path(output_dir)

    images_dir = yolo_path / "images"
    labels_dir = yolo_path / "labels"

    # Read train.csv to get video_id mapping
    train_df = pd.read_csv("data/train.csv")

    # Get unique videos
    videos = sorted(train_df["video_id"].unique())
    print(f"Found {len(videos)} videos: {videos}")

    # Create 3 folds (leave-one-video-out)
    folds = [
        {"train": [1, 2], "val": [0]},  # Fold 0
        {"train": [0, 2], "val": [1]},  # Fold 1
        {"train": [0, 1], "val": [2]},  # Fold 2
    ]

    for fold_idx, fold_config in enumerate(folds):
        print(f"\nCreating Fold {fold_idx}...")
        print(f"  Train: video_{fold_config['train']}")
        print(f"  Val: video_{fold_config['val']}")

        # Create fold directories
        fold_dir = output_path / f"fold_{fold_idx}"
        for split in ["train", "val"]:
            (fold_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Get image IDs for this fold
        train_ids = train_df[train_df["video_id"].isin(fold_config["train"])]["image_id"].tolist()
        val_ids = train_df[train_df["video_id"].isin(fold_config["val"])]["image_id"].tolist()

        # Copy files for train split
        print(f"  Copying {len(train_ids)} train images...")
        for image_id in tqdm(train_ids):
            src_img = images_dir / f"{image_id}.jpg"
            src_lbl = labels_dir / f"{image_id}.txt"
            dst_img = fold_dir / "train" / "images" / f"{image_id}.jpg"
            dst_lbl = fold_dir / "train" / "labels" / f"{image_id}.txt"

            if src_img.exists():
                shutil.copy(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy(src_lbl, dst_lbl)

        # Copy files for val split
        print(f"  Copying {len(val_ids)} val images...")
        for image_id in tqdm(val_ids):
            src_img = images_dir / f"{image_id}.jpg"
            src_lbl = labels_dir / f"{image_id}.txt"
            dst_img = fold_dir / "val" / "images" / f"{image_id}.jpg"
            dst_lbl = fold_dir / "val" / "labels" / f"{image_id}.txt"

            if src_img.exists():
                shutil.copy(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy(src_lbl, dst_lbl)

        print(f"✓ Fold {fold_idx} complete!")

    print(f"\n✓ All folds created in: {output_path}")


def main():
    """Main function."""
    create_fold_splits()


if __name__ == "__main__":
    main()
