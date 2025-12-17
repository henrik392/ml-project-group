"""
Data Analysis for TensorFlow Great Barrier Reef Competition
Analyzes the training data, annotations, and image distribution
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter, defaultdict
import ast

# Paths
DATA_DIR = Path("data")
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
TRAIN_IMAGES = DATA_DIR / "train_images"

def analyze_csv_structure():
    """Analyze the structure of train and test CSV files"""
    print("=" * 80)
    print("CSV STRUCTURE ANALYSIS")
    print("=" * 80)

    # Load data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print(f"\n📊 Train CSV Shape: {train_df.shape}")
    print(f"📊 Test CSV Shape: {test_df.shape}")

    print(f"\n📋 Train Columns: {list(train_df.columns)}")
    print(f"📋 Test Columns: {list(test_df.columns)}")

    print(f"\n📊 Train Data Info:")
    print(train_df.info())

    print(f"\n📊 First few rows of train data:")
    print(train_df.head(10))

    return train_df, test_df

def analyze_videos_and_sequences(train_df):
    """Analyze video and sequence distribution"""
    print("\n" + "=" * 80)
    print("VIDEO & SEQUENCE ANALYSIS")
    print("=" * 80)

    # Video statistics
    n_videos = train_df['video_id'].nunique()
    n_sequences = train_df['sequence'].nunique()

    print(f"\n📹 Total unique videos: {n_videos}")
    print(f"📹 Total unique sequences: {n_sequences}")
    print(f"📹 Total frames: {len(train_df)}")

    # Frames per video
    frames_per_video = train_df.groupby('video_id').size()
    print(f"\n📊 Frames per video:")
    print(f"  - Mean: {frames_per_video.mean():.2f}")
    print(f"  - Median: {frames_per_video.median():.2f}")
    print(f"  - Min: {frames_per_video.min()}")
    print(f"  - Max: {frames_per_video.max()}")

    # Frames per sequence
    frames_per_sequence = train_df.groupby('sequence').size()
    print(f"\n📊 Frames per sequence:")
    print(f"  - Mean: {frames_per_sequence.mean():.2f}")
    print(f"  - Median: {frames_per_sequence.median():.2f}")
    print(f"  - Min: {frames_per_sequence.min()}")
    print(f"  - Max: {frames_per_sequence.max()}")

    # Video distribution
    print(f"\n📊 Top 10 videos by frame count:")
    print(frames_per_video.sort_values(ascending=False).head(10))

    return {
        'n_videos': n_videos,
        'n_sequences': n_sequences,
        'frames_per_video': frames_per_video,
        'frames_per_sequence': frames_per_sequence
    }

def analyze_annotations(train_df):
    """Analyze the annotations (bounding boxes)"""
    print("\n" + "=" * 80)
    print("ANNOTATION ANALYSIS")
    print("=" * 80)

    # Parse annotations
    def parse_annotation(ann_str):
        """Parse annotation string to list of dicts"""
        try:
            return ast.literal_eval(ann_str)
        except:
            return []

    train_df['parsed_annotations'] = train_df['annotations'].apply(parse_annotation)
    train_df['n_objects'] = train_df['parsed_annotations'].apply(len)

    # Basic statistics
    total_frames = len(train_df)
    frames_with_objects = (train_df['n_objects'] > 0).sum()
    frames_without_objects = (train_df['n_objects'] == 0).sum()
    total_objects = train_df['n_objects'].sum()

    print(f"\n🎯 Object Detection Statistics:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Frames with objects: {frames_with_objects} ({frames_with_objects/total_frames*100:.2f}%)")
    print(f"  - Frames without objects: {frames_without_objects} ({frames_without_objects/total_frames*100:.2f}%)")
    print(f"  - Total starfish annotations: {total_objects}")
    print(f"  - Average objects per frame: {train_df['n_objects'].mean():.3f}")
    print(f"  - Average objects per frame (with objects): {train_df[train_df['n_objects'] > 0]['n_objects'].mean():.3f}")

    # Object count distribution
    print(f"\n📊 Object count distribution:")
    object_counts = train_df['n_objects'].value_counts().sort_index()
    for count, freq in object_counts.items():
        print(f"  - {count} objects: {freq} frames ({freq/total_frames*100:.2f}%)")

    # Bounding box statistics
    all_boxes = []
    for anns in train_df['parsed_annotations']:
        all_boxes.extend(anns)

    if all_boxes:
        widths = [box['width'] for box in all_boxes]
        heights = [box['height'] for box in all_boxes]
        areas = [box['width'] * box['height'] for box in all_boxes]
        aspect_ratios = [box['width'] / box['height'] if box['height'] > 0 else 0 for box in all_boxes]

        print(f"\n📏 Bounding Box Statistics (n={len(all_boxes)}):")
        print(f"\n  Width:")
        print(f"    - Mean: {np.mean(widths):.2f}")
        print(f"    - Median: {np.median(widths):.2f}")
        print(f"    - Min: {np.min(widths)}")
        print(f"    - Max: {np.max(widths)}")

        print(f"\n  Height:")
        print(f"    - Mean: {np.mean(heights):.2f}")
        print(f"    - Median: {np.median(heights):.2f}")
        print(f"    - Min: {np.min(heights)}")
        print(f"    - Max: {np.max(heights)}")

        print(f"\n  Area:")
        print(f"    - Mean: {np.mean(areas):.2f}")
        print(f"    - Median: {np.median(areas):.2f}")
        print(f"    - Min: {np.min(areas)}")
        print(f"    - Max: {np.max(areas)}")

        print(f"\n  Aspect Ratio (width/height):")
        print(f"    - Mean: {np.mean(aspect_ratios):.2f}")
        print(f"    - Median: {np.median(aspect_ratios):.2f}")
        print(f"    - Min: {np.min(aspect_ratios):.2f}")
        print(f"    - Max: {np.max(aspect_ratios):.2f}")

    return train_df, {
        'total_objects': total_objects,
        'frames_with_objects': frames_with_objects,
        'all_boxes': all_boxes
    }

def analyze_temporal_patterns(train_df):
    """Analyze temporal patterns in annotations"""
    print("\n" + "=" * 80)
    print("TEMPORAL PATTERN ANALYSIS")
    print("=" * 80)

    # Group by video and sequence
    video_groups = train_df.groupby('video_id')

    print(f"\n🎬 Object presence by video:")
    for video_id, group in video_groups:
        frames_with_objects = (group['n_objects'] > 0).sum()
        total_frames = len(group)
        total_objects = group['n_objects'].sum()
        print(f"  Video {video_id}: {frames_with_objects}/{total_frames} frames with objects ({frames_with_objects/total_frames*100:.1f}%), {total_objects} total objects")

    # Analyze consecutive frames with objects
    print(f"\n🔄 Temporal continuity:")
    for video_id, group in video_groups:
        group = group.sort_values('video_frame')
        has_objects = (group['n_objects'] > 0).astype(int).values

        # Find runs of frames with objects
        changes = np.diff(np.concatenate([[0], has_objects, [0]]))
        run_starts = np.where(changes == 1)[0]
        run_ends = np.where(changes == -1)[0]
        run_lengths = run_ends - run_starts

        if len(run_lengths) > 0:
            print(f"  Video {video_id}:")
            print(f"    - Object sequences: {len(run_lengths)}")
            print(f"    - Avg sequence length: {np.mean(run_lengths):.1f} frames")
            print(f"    - Max sequence length: {np.max(run_lengths)} frames")

def analyze_image_distribution():
    """Analyze the distribution of images in the train_images directory"""
    print("\n" + "=" * 80)
    print("IMAGE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    if not TRAIN_IMAGES.exists():
        print("❌ train_images directory not found")
        return

    video_dirs = sorted([d for d in TRAIN_IMAGES.iterdir() if d.is_dir()])

    print(f"\n📁 Found {len(video_dirs)} video directories")

    for video_dir in video_dirs:
        image_files = list(video_dir.glob("*.jpg"))
        print(f"  {video_dir.name}: {len(image_files)} images")

def generate_summary_stats(train_df, video_stats, annotation_stats):
    """Generate summary statistics"""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"""
🎯 Dataset Overview:
   - Total training frames: {len(train_df):,}
   - Unique videos: {video_stats['n_videos']}
   - Unique sequences: {video_stats['n_sequences']}
   - Total starfish annotations: {annotation_stats['total_objects']:,}

📊 Class Imbalance:
   - Positive frames (with starfish): {annotation_stats['frames_with_objects']:,} ({annotation_stats['frames_with_objects']/len(train_df)*100:.2f}%)
   - Negative frames (no starfish): {len(train_df) - annotation_stats['frames_with_objects']:,} ({(len(train_df) - annotation_stats['frames_with_objects'])/len(train_df)*100:.2f}%)
   - Imbalance ratio: 1:{(len(train_df) - annotation_stats['frames_with_objects'])/annotation_stats['frames_with_objects']:.2f}

🎬 Temporal Structure:
   - Frames are organized as sequential video frames
   - Multiple sequences per video
   - Object tracking across consecutive frames possible

🔍 Key Observations:
   - This is a VIDEO object detection task (temporal information available)
   - Highly imbalanced dataset (most frames have no starfish)
   - Bounding boxes vary significantly in size
   - Starfish can appear and move across frames
    """)

def main():
    """Main analysis function"""
    print("\n" + "=" * 80)
    print("TENSORFLOW GREAT BARRIER REEF - DATA ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing Crown-of-Thorns Starfish (COTS) Detection Dataset")

    # Run analyses
    train_df, test_df = analyze_csv_structure()
    video_stats = analyze_videos_and_sequences(train_df)
    train_df, annotation_stats = analyze_annotations(train_df)
    analyze_temporal_patterns(train_df)
    analyze_image_distribution()
    generate_summary_stats(train_df, video_stats, annotation_stats)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
