"""
Inference utilities for COTS detection.

Supports:
- Standard inference
- SAHI tiled inference (for small objects)
- ByteTrack tracking (for temporal context)
"""

import time
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


def load_frames(source: str) -> tuple[list, list]:
    """
    Load frames from video file or image directory.

    Returns:
        Tuple of (frames, frame_ids) where frame_ids are based on filename (without extension)
    """
    source_path = Path(source)

    if source_path.is_file():
        # Video file - use sequential frame numbers
        cap = cv2.VideoCapture(str(source_path))
        frames = []
        frame_ids = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_ids.append(str(idx))
            idx += 1
        cap.release()
    elif source_path.is_dir():
        # Directory of images - use filename (without extension) as frame ID
        paths = sorted(
            list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        )
        frames = [cv2.imread(str(p)) for p in paths]
        frame_ids = [p.stem for p in paths]  # Use filename without extension
    else:
        raise ValueError(f"Invalid source: {source}")

    return frames, frame_ids


def to_tracker_format(detections, detection_type: str) -> np.ndarray:
    """
    Convert detections to tracker format (Nx6: [x1, y1, x2, y2, score, class]).

    Args:
        detections: YOLO Results object or SAHI object_prediction_list
        detection_type: 'yolo' or 'sahi'

    Returns:
        Nx6 numpy array for tracker
    """
    rows = []

    if detection_type == "sahi":
        for pred in detections:
            x1, y1, x2, y2 = pred.bbox.to_voc_bbox()
            rows.append(
                [x1, y1, x2, y2, float(pred.score.value), int(pred.category.id)]
            )
    else:  # yolo
        if detections.boxes is not None and len(detections.boxes) > 0:
            for box in detections.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                rows.append([x1, y1, x2, y2, conf, cls])

    return (
        np.asarray(rows, dtype=np.float32)
        if rows
        else np.zeros((0, 6), dtype=np.float32)
    )


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

    # Load frames
    frames, frame_ids = load_frames(source)
    if verbose:
        print(f"Loaded {len(frames)} frames from {source}")

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

    # Initialize tracker (optional)
    tracker = None
    if use_tracking:
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(f"{tracker_name}.yaml")))
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        if verbose:
            print(f"Initialized {tracker_name} tracker")

    results = []
    total_time = 0

    for idx, frame in enumerate(frames):
        start = time.time()

        # Step 1: Detect
        if use_sahi:
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
            detections = det_result.object_prediction_list
        else:
            det_result = detector.predict(
                frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False
            )[0]
            detections = det_result

        # Step 2: Track (optional)
        if use_tracking:
            dets_array = to_tracker_format(detections, "sahi" if use_sahi else "yolo")
            tracks = tracker.update(dets_array, frame)
            result = {
                "frame_idx": idx,
                "detections": det_result,
                "tracks": tracks,
            }
        else:
            result = det_result

        results.append(result)
        elapsed = time.time() - start
        total_time += elapsed

        if verbose and (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(frames)} frames ({elapsed * 1000:.1f}ms)")

    ms_per_frame = (total_time / len(frames)) * 1000 if frames else 0
    return {
        "predictions": results,
        "frame_ids": frame_ids,
        "ms_per_frame": ms_per_frame,
        "num_images": len(results),
    }
