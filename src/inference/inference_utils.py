"""
Inference utilities for COTS detection.

Supports:
- Standard inference
- SAHI tiled inference (for small objects)
- ByteTrack tracking (for temporal context)

Note: SAHI and ByteTrack are mutually exclusive modes.
"""

import gc
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


def get_frame_sources(source: str) -> tuple[Path, list, list]:
    """
    Get frame sources without loading frames into memory.

    Returns:
        Tuple of (source_path, frame_sources, frame_ids)
        - source_path: Path object for the source
        - frame_sources: List of indices (for video) or Paths (for images)
        - frame_ids: List of frame IDs (str)
    """
    source_path = Path(source)

    if source_path.is_file():
        # Video file - return frame count and indices
        cap = cv2.VideoCapture(str(source_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_sources = list(range(frame_count))
        frame_ids = [str(i) for i in range(frame_count)]
    elif source_path.is_dir():
        # Directory of images - return sorted paths
        frame_sources = sorted(
            list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        )
        frame_ids = [p.stem for p in frame_sources]
    else:
        raise ValueError(f"Invalid source: {source}")

    return source_path, frame_sources, frame_ids


def load_single_frame(source_path: Path, frame_source) -> np.ndarray:
    """
    Load a single frame from video or image.

    Args:
        source_path: Path to video file or image directory
        frame_source: Frame index (int) for video, or Path for image

    Returns:
        Frame as numpy array (BGR format)
    """
    if source_path.is_file():
        # Video file - seek to frame index
        cap = cv2.VideoCapture(str(source_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_source)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Failed to read frame {frame_source} from {source_path}")
        return frame
    else:
        # Image file
        frame = cv2.imread(str(frame_source))
        if frame is None:
            raise ValueError(f"Failed to read image {frame_source}")
        return frame


def run_inference(config: dict, weights_path: str) -> dict:
    """
    Unified inference pipeline from experiment config.

    Supports:
    - Standard inference
    - SAHI tiled inference
    - ByteTrack tracking

    Pipeline:
    1. Load frames from source
    2. Detect per frame (SAHI or YOLO)
    3. Track across frames (optional)
    4. Return results + latency

    Args:
        config: Experiment config with inference settings
        weights_path: Path to model weights

    Returns:
        Dict with 'predictions', 'ms_per_frame', 'num_images'
    """
    # Extract config values
    inference_config = config.get("inference", {})
    tracking_config = config.get("tracking", {})

    mode = inference_config.get("mode", "standard")
    conf = inference_config.get("conf", 0.25)
    iou = inference_config.get("iou", 0.45)
    imgsz = config.get("imgsz", 640)
    device = config.get("device", "mps")

    # Determine source
    fold_id = config.get("fold_id", 0)
    eval_video_id = config.get("eval_video_id", f"video_{fold_id}")
    source = f"data/train_images/{eval_video_id}"

    # Setup flags
    use_sahi = mode == "sahi"
    use_tracking = tracking_config.get("enabled", False)
    sahi_config = inference_config.get("sahi")
    tracker_name = tracking_config.get("tracker", "bytetrack")
    verbose = True

    print(f"[INFO] Running {'SAHI' if use_sahi else 'Standard'}", end="")
    print(" + ByteTrack" if use_tracking else "", "inference...")

    # Get frame sources without loading all frames
    source_path, frame_sources, frame_ids = get_frame_sources(source)
    if verbose:
        print(f"Processing {len(frame_sources)} frames from {source}")

    # Initialize detector
    if use_sahi:
        detector = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=weights_path,
            confidence_threshold=conf,
            device=device,
        )
        sahi_cfg = sahi_config or {}
    else:
        detector = YOLO(weights_path)

    results = []
    total_time = 0

    # Determine tracker config path
    tracker_config = None
    if use_tracking:
        # Use custom tracker config if it exists, otherwise use default
        custom_tracker = "configs/trackers/bytetrack_custom.yaml"
        if Path(custom_tracker).exists():
            tracker_config = custom_tracker
        else:
            tracker_config = f"{tracker_name}.yaml"

    # Open video file once if source is a video (for performance)
    video_cap = None
    if source_path.is_file():
        video_cap = cv2.VideoCapture(str(source_path))

    try:
        # Process frames one at a time
        for idx, frame_source in enumerate(frame_sources):
            # Load single frame
            if video_cap:
                ret, frame = video_cap.read()
                if not ret:
                    raise ValueError(f"Failed to read frame {idx} from video")
            else:
                frame = load_single_frame(source_path, frame_source)

            start = time.time()

            # Run detection or tracking
            if use_sahi:
                # SAHI inference (no tracking support)
                det_result = get_sliced_prediction(
                    frame,
                    detector,
                    slice_height=sahi_cfg.get("slice_height", 640),
                    slice_width=sahi_cfg.get("slice_width", 640),
                    overlap_height_ratio=sahi_cfg.get("overlap_height_ratio", 0.2),
                    overlap_width_ratio=sahi_cfg.get("overlap_width_ratio", 0.2),
                    postprocess_type=sahi_cfg.get("postprocess_type", "NMS"),
                    postprocess_match_metric=sahi_cfg.get(
                        "postprocess_match_metric", "IOS"
                    ),
                    postprocess_match_threshold=sahi_cfg.get(
                        "postprocess_match_threshold", 0.5
                    ),
                    verbose=0,
                )
            elif use_tracking:
                # Use official model.track() - maintains state with persist=True
                det_result = detector.track(
                    frame,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    persist=True,
                    tracker=tracker_config,
                    verbose=False,
                )[0]
            else:
                # Standard detection
                det_result = detector.predict(
                    frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False
                )[0]

            results.append(det_result)
            elapsed = time.time() - start
            total_time += elapsed

            # Release frame memory
            del frame

            # For SAHI: aggressive memory cleanup after each frame
            if use_sahi:
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if verbose and (idx + 1) % 10 == 0:
                print(
                    f"Processed {idx + 1}/{len(frame_sources)} frames ({elapsed * 1000:.1f}ms)"
                )
    finally:
        # Always close video capture if opened
        if video_cap:
            video_cap.release()

    ms_per_frame = (total_time / len(frame_sources)) * 1000 if frame_sources else 0
    return {
        "predictions": results,
        "frame_ids": frame_ids,
        "ms_per_frame": ms_per_frame,
        "num_images": len(results),
    }
