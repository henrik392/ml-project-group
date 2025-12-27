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
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml


def run_standard_inference(
    model_path: str,
    source: str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "mps",
    verbose: bool = False,
) -> tuple[list, float]:
    """
    Run standard YOLO inference.

    Args:
        model_path: Path to model weights
        source: Image path or directory
        conf: Confidence threshold
        iou: IoU threshold for NMS
        imgsz: Image size
        device: Device to use
        verbose: Print verbose output

    Returns:
        Tuple of (predictions list, average ms per frame)
    """
    model = YOLO(model_path)

    start_time = time.time()
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=verbose,
    )
    elapsed = time.time() - start_time

    # Calculate average latency
    num_images = len(results) if isinstance(results, list) else 1
    ms_per_frame = (elapsed / num_images) * 1000 if num_images > 0 else 0

    return results, ms_per_frame


def run_sahi_inference(
    model_path: str,
    source: str,
    conf: float = 0.25,
    iou: float = 0.45,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "NMS",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[list, float]:
    """
    Run SAHI tiled inference for small object detection.

    Args:
        model_path: Path to model weights
        source: Image path or directory
        conf: Confidence threshold
        iou: IoU threshold
        slice_height: Height of each slice
        slice_width: Width of each slice
        overlap_height_ratio: Overlap ratio for height
        overlap_width_ratio: Overlap ratio for width
        postprocess_type: NMS or GREEDYNMM
        postprocess_match_metric: IOS or IOU
        postprocess_match_threshold: Threshold for matching
        device: Device to use ('cpu' or 'cuda:0')
        verbose: Print verbose output

    Returns:
        Tuple of (predictions list, average ms per frame)
    """
    # Initialize SAHI detection model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )

    # Get list of images
    source_path = Path(source)
    if source_path.is_file():
        image_paths = [source_path]
    elif source_path.is_dir():
        image_paths = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
    else:
        raise ValueError(f"Invalid source: {source}")

    results = []
    total_time = 0

    for image_path in image_paths:
        start_time = time.time()

        result = get_sliced_prediction(
            str(image_path),
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_type=postprocess_type,
            postprocess_match_metric=postprocess_match_metric,
            postprocess_match_threshold=postprocess_match_threshold,
            verbose=0 if not verbose else 1,
        )

        elapsed = time.time() - start_time
        total_time += elapsed

        results.append(result)

        if verbose:
            print(f"Processed {image_path.name} in {elapsed * 1000:.1f}ms")

    # Calculate average latency
    num_images = len(results)
    ms_per_frame = (total_time / num_images) * 1000 if num_images > 0 else 0

    return results, ms_per_frame


def run_tracking_inference(
    model_path: str,
    source: str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "mps",
    tracker: str = "bytetrack",
    tracker_config: str = None,
    verbose: bool = False,
) -> tuple[list, float]:
    """
    Run inference with ByteTrack tracking for temporal context.

    Args:
        model_path: Path to model weights
        source: Video path or directory of frames
        conf: Confidence threshold
        iou: IoU threshold
        imgsz: Image size
        device: Device to use
        tracker: Tracker name ('bytetrack' or 'botsort')
        tracker_config: Path to custom tracker config YAML
        verbose: Print verbose output

    Returns:
        Tuple of (tracking results list, average ms per frame)
    """
    model = YOLO(model_path)

    start_time = time.time()

    # Use model.track() instead of model.predict()
    if tracker_config:
        results = model.track(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            tracker=tracker_config,
            persist=True,
            verbose=verbose,
        )
    else:
        # Use default tracker config
        tracker_yaml = f"{tracker}.yaml"
        results = model.track(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            tracker=tracker_yaml,
            persist=True,
            verbose=verbose,
        )

    elapsed = time.time() - start_time

    # Calculate average latency
    num_frames = len(results) if isinstance(results, list) else 1
    ms_per_frame = (elapsed / num_frames) * 1000 if num_frames > 0 else 0

    return results, ms_per_frame


def sahi_to_tracker_format(sahi_predictions) -> np.ndarray:
    """
    Convert SAHI predictions to tracker format.

    Args:
        sahi_predictions: List of SAHI ObjectPrediction objects

    Returns:
        Nx6 numpy array: [x1, y1, x2, y2, score, class]
    """
    rows = []
    for pred in sahi_predictions:
        x1, y1, x2, y2 = pred.bbox.to_voc_bbox()
        rows.append(
            [
                x1,
                y1,
                x2,
                y2,
                float(pred.score.value),
                int(pred.category.id),
            ]
        )
    return (
        np.asarray(rows, dtype=np.float32)
        if rows
        else np.zeros((0, 6), dtype=np.float32)
    )


def run_sahi_tracking_inference(
    model_path: str,
    source: str,
    conf: float = 0.15,
    iou: float = 0.45,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "NMS",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    tracker: str = "bytetrack",
    tracker_config: str = None,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[list, float]:
    """
    Run SAHI sliced detection + ByteTrack tracking combined.

    Pipeline:
    1. SAHI sliced detection per frame → merged global boxes
    2. Convert to tracker format (Nx6: [x1, y1, x2, y2, score, class])
    3. ByteTrack tracker.update() for temporal consistency
    4. Return tracked results with IDs

    Args:
        model_path: Path to model weights
        source: Video path or directory of frames
        conf: Confidence threshold (use low 0.10-0.25 for tracker)
        iou: IoU threshold
        slice_height: Height of each slice
        slice_width: Width of each slice
        overlap_height_ratio: Overlap ratio for height
        overlap_width_ratio: Overlap ratio for width
        postprocess_type: NMS or GREEDYNMM
        postprocess_match_metric: IOS or IOU
        postprocess_match_threshold: Threshold for matching
        tracker: Tracker name ('bytetrack' or 'botsort')
        tracker_config: Path to custom tracker config YAML
        device: Device to use ('cpu' or 'cuda:0')
        verbose: Print verbose output

    Returns:
        Tuple of (tracking results list, average ms per frame)
    """
    # Initialize SAHI detection model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )

    # Initialize tracker
    if tracker_config:
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_config)))
    else:
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(f"{tracker}.yaml")))

    tracker_instance = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)

    # Get list of frames
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
        image_paths = sorted(
            list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        )
        frames = [cv2.imread(str(img_path)) for img_path in image_paths]
    else:
        raise ValueError(f"Invalid source: {source}")

    results = []
    total_time = 0

    for idx, frame in enumerate(frames):
        start_time = time.time()

        # 1. SAHI sliced detection
        sahi_result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_type=postprocess_type,
            postprocess_match_metric=postprocess_match_metric,
            postprocess_match_threshold=postprocess_match_threshold,
            verbose=0 if not verbose else 1,
        )

        # 2. Convert SAHI predictions to tracker format
        dets = sahi_to_tracker_format(sahi_result.object_prediction_list)

        # 3. Update tracker with merged detections
        tracks = tracker_instance.update(dets, frame)

        elapsed = time.time() - start_time
        total_time += elapsed

        # Store result
        results.append(
            {
                "frame_idx": idx,
                "tracks": tracks,
                "detections": dets,
                "sahi_result": sahi_result,
            }
        )

        if verbose:
            print(
                f"Frame {idx + 1}/{len(frames)}: {len(dets)} dets → {len(tracks) if tracks is not None else 0} tracks ({elapsed * 1000:.1f}ms)"
            )

    # Calculate average latency
    num_frames = len(results)
    ms_per_frame = (total_time / num_frames) * 1000 if num_frames > 0 else 0

    return results, ms_per_frame


def run_inference_from_config(
    config: dict,
    weights_path: str,
) -> dict:
    """
    Run inference based on experiment config.

    Args:
        config: Experiment config with inference settings
        weights_path: Path to model weights

    Returns:
        Dict with predictions and latency metrics
    """
    inference_config = config.get("inference", {})
    tracking_config = config.get("tracking", {})

    mode = inference_config.get("mode", "standard")
    conf = inference_config.get("conf", 0.25)
    iou = inference_config.get("iou", 0.45)
    imgsz = config.get("imgsz", 640)
    device = config.get("device", "mps")

    # Determine source (for now, use validation set)
    fold_id = config.get("fold_id", 0)
    eval_video_id = config.get("eval_video_id", f"video_{fold_id}")
    source = f"data/train_images/{eval_video_id}"

    print(f"\n{'=' * 80}")
    print(f"Running {mode} inference on {eval_video_id}")
    print(f"conf={conf}, iou={iou}, imgsz={imgsz}")
    print(f"{'=' * 80}\n")

    # Run inference based on mode
    tracking_enabled = tracking_config.get("enabled", False)

    if mode == "sahi" and tracking_enabled:
        # Combined SAHI + ByteTrack mode
        print("Mode: SAHI + ByteTrack (combined)")
        sahi_config = inference_config.get("sahi", {})
        tracker = tracking_config.get("tracker", "bytetrack")
        results, ms_per_frame = run_sahi_tracking_inference(
            model_path=weights_path,
            source=source,
            conf=conf,
            iou=iou,
            slice_height=sahi_config.get("slice_height", 640),
            slice_width=sahi_config.get("slice_width", 640),
            overlap_height_ratio=sahi_config.get("overlap_height_ratio", 0.2),
            overlap_width_ratio=sahi_config.get("overlap_width_ratio", 0.2),
            postprocess_type=sahi_config.get("postprocess_type", "NMS"),
            postprocess_match_metric=sahi_config.get("postprocess_match_metric", "IOS"),
            postprocess_match_threshold=sahi_config.get(
                "postprocess_match_threshold", 0.5
            ),
            tracker=tracker,
            device="cpu" if device == "mps" else device,
            verbose=True,
        )
    elif mode == "sahi":
        # SAHI inference only
        print("Mode: SAHI (tiled inference)")
        sahi_config = inference_config.get("sahi", {})
        results, ms_per_frame = run_sahi_inference(
            model_path=weights_path,
            source=source,
            conf=conf,
            iou=iou,
            slice_height=sahi_config.get("slice_height", 640),
            slice_width=sahi_config.get("slice_width", 640),
            overlap_height_ratio=sahi_config.get("overlap_height_ratio", 0.2),
            overlap_width_ratio=sahi_config.get("overlap_width_ratio", 0.2),
            postprocess_type=sahi_config.get("postprocess_type", "NMS"),
            postprocess_match_metric=sahi_config.get("postprocess_match_metric", "IOS"),
            postprocess_match_threshold=sahi_config.get(
                "postprocess_match_threshold", 0.5
            ),
            device="cpu" if device == "mps" else device,
            verbose=True,
        )
    elif tracking_enabled:
        # ByteTrack tracking only
        print("Mode: ByteTrack (temporal tracking)")
        tracker = tracking_config.get("tracker", "bytetrack")
        results, ms_per_frame = run_tracking_inference(
            model_path=weights_path,
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            tracker=tracker,
            verbose=True,
        )
    else:
        # Standard inference
        print("Mode: Standard (no SAHI, no tracking)")
        results, ms_per_frame = run_standard_inference(
            model_path=weights_path,
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=True,
        )

    print(f"\nInference complete: {ms_per_frame:.2f} ms/frame")

    return {
        "predictions": results,
        "ms_per_frame": ms_per_frame,
        "num_images": len(results) if isinstance(results, list) else 1,
    }
