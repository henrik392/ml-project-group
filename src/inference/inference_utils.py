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


def load_frames(source: str) -> list:
    """Load frames from video file or image directory."""
    source_path = Path(source)

    if source_path.is_file():
        # Video file
        cap = cv2.VideoCapture(str(source_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    elif source_path.is_dir():
        # Directory of images
        paths = sorted(
            list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        )
        frames = [cv2.imread(str(p)) for p in paths]
    else:
        raise ValueError(f"Invalid source: {source}")

    return frames


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


def run_inference(
    model_path: str,
    source: str,
    use_sahi: bool = False,
    use_tracking: bool = False,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    sahi_config: dict = None,
    tracker_name: str = "bytetrack",
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[list, float]:
    """
    Unified inference pipeline with optional SAHI and tracking.

    Pipeline:
    1. Load frames from source
    2. Detect per frame (SAHI or YOLO)
    3. Track across frames (optional)
    4. Return results + latency

    Args:
        model_path: Path to model weights
        source: Video path or image directory
        use_sahi: Use SAHI tiled inference
        use_tracking: Use ByteTrack tracking
        conf: Confidence threshold
        iou: IoU threshold (YOLO only)
        imgsz: Image size (YOLO only)
        sahi_config: SAHI configuration dict
        tracker_name: Tracker name ('bytetrack' or 'botsort')
        device: Device ('cpu', 'cuda:0', 'mps')
        verbose: Print progress

    Returns:
        Tuple of (results list, ms per frame)
    """
    # Load frames
    frames = load_frames(source)
    if verbose:
        print(f"Loaded {len(frames)} frames from {source}")

    # Initialize detector
    if use_sahi:
        detector = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=conf,
            device=device,
        )
        sahi_cfg = sahi_config or {}
    else:
        detector = YOLO(model_path)

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
    return results, ms_per_frame
