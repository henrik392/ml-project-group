"""
Generate 2x2 comparison video of COTS detection methods.

Creates a side-by-side visualization comparing:
- YOLOv5 baseline
- YOLOv11 baseline
- SAHI tiled inference
- SAHI + ByteTrack
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.inference.inference_utils import load_frames, run_inference
from src.run_experiment import load_config


@dataclass
class MethodConfig:
    """Configuration for a detection method in the grid."""

    name: str
    experiment_id: str
    config_path: str
    color: tuple[int, int, int]


METHODS = [
    MethodConfig(
        name="YOLOv5",
        experiment_id="exp01_yolov5",
        config_path="configs/experiments/exp01_yolov5_baseline.yaml",
        color=(0, 165, 255),  # Orange
    ),
    MethodConfig(
        name="YOLOv11",
        experiment_id="exp02_yolov11",
        config_path="configs/experiments/exp02_yolov11_baseline.yaml",
        color=(0, 255, 0),  # Green
    ),
    MethodConfig(
        name="SAHI",
        experiment_id="exp06_sahi",
        config_path="configs/experiments/exp06_sahi_inference.yaml",
        color=(255, 0, 0),  # Blue
    ),
    MethodConfig(
        name="SAHI+ByteTrack",
        experiment_id="exp08_sahi_bytetrack",
        config_path="configs/experiments/exp08_sahi_bytetrack.yaml",
        color=(255, 0, 255),  # Magenta
    ),
]


def convert_predictions_to_list(results: list, has_tracking: bool) -> list[list[list]]:
    """
    Convert YOLO/SAHI/tracking results to serializable list format.

    Args:
        results: List of YOLO Results, SAHI PredictionResult, or tracking dicts
        has_tracking: Whether results include tracking information

    Returns:
        List of detections per frame.
        Each frame has list of [x1, y1, x2, y2, conf, cls] lists.
    """
    all_detections = []

    for result in results:
        frame_detections = []

        # Extract detection object (handle tracking dict format)
        if has_tracking:
            det_result = result["detections"]
        else:
            det_result = result

        # SAHI format
        if hasattr(det_result, "object_prediction_list"):
            for obj_pred in det_result.object_prediction_list:
                bbox = obj_pred.bbox
                frame_detections.append(
                    [
                        float(bbox.minx),
                        float(bbox.miny),
                        float(bbox.maxx),
                        float(bbox.maxy),
                        float(obj_pred.score.value),
                        0,
                    ]
                )
        # YOLO format
        elif hasattr(det_result, "boxes") and det_result.boxes is not None:
            for box in det_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                frame_detections.append(
                    [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                        float(box.conf[0]),
                        int(box.cls[0]),
                    ]
                )

        all_detections.append(frame_detections)

    return all_detections


def load_or_run_inference(
    method: MethodConfig,
    video_id: str,
    cache_dir: Path,
    force_rerun: bool = False,
    frames: list | None = None,
) -> list[list[list]]:
    """
    Load cached predictions or run inference.

    Args:
        method: Method configuration
        video_id: Video to process (e.g., 'video_0')
        cache_dir: Directory for prediction cache
        force_rerun: If True, rerun inference even if cached
        frames: Pre-loaded frames to use (for limited testing)

    Returns:
        List of detections per frame.
    """
    # Use different cache key if using limited frames
    if frames is not None:
        cache_suffix = f"_{len(frames)}frames"
    else:
        cache_suffix = ""
    cache_path = cache_dir / f"{method.experiment_id}_{video_id}{cache_suffix}.json"

    if cache_path.exists() and not force_rerun:
        print(f"  Loading from cache: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    # Load config
    config = load_config(method.config_path)
    config["eval_video_id"] = video_id

    # Determine weights path
    if config.get("skip_training") and config.get("weights"):
        # Use weights specified in config (for inference-only experiments)
        weights_path = Path(config["weights"])
    else:
        # Construct path from model name and fold (for training experiments)
        model_name = config.get("model", "yolov11n")
        # Normalize model name (yolov11n -> yolo11n for directory name, but keep yolov5n as is)
        if "yolov11" in model_name:
            model_name = model_name.replace("yolov11", "yolo11")
        fold_id = config.get("fold_id", 0)
        weights_path = Path(f"runs/train/{model_name}_fold{fold_id}/weights/best.pt")

    if not weights_path.exists():
        print(f"\nERROR: Weights not found for {method.name}")
        print(f"  Expected: {weights_path}")
        print(f"  Config: {method.config_path}")
        print("\nPlease run the experiment to train the model first:")
        print(f"  uv run python -m src.run_experiment {method.config_path}")
        sys.exit(1)

    # Run inference
    if frames is not None:
        # Run inference on pre-loaded frames (for testing)
        print(f"  Running inference on {len(frames)} frames...")
        from ultralytics import YOLO
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        device = config.get("device", "mps")
        conf = config.get("inference", {}).get("conf", 0.25)
        iou = config.get("inference", {}).get("iou", 0.45)
        imgsz = config.get("imgsz", 640)
        use_sahi = config.get("inference", {}).get("mode") == "sahi"

        # Load model
        if use_sahi:
            detector = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(weights_path),
                confidence_threshold=conf,
                device=device,
            )
        else:
            detector = YOLO(str(weights_path))

        # Run detection on each frame
        frame_results = []
        for idx, frame in enumerate(frames):
            if use_sahi:
                sahi_config = config.get("inference", {}).get("sahi", {})
                det_result = get_sliced_prediction(
                    frame,
                    detector,
                    slice_height=sahi_config.get("slice_height", 640),
                    slice_width=sahi_config.get("slice_width", 640),
                    overlap_height_ratio=sahi_config.get("overlap_height_ratio", 0.2),
                    overlap_width_ratio=sahi_config.get("overlap_width_ratio", 0.2),
                    verbose=0,
                )
            else:
                det_result = detector.predict(
                    frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False
                )[0]
            frame_results.append(det_result)
            if (idx + 1) % 10 == 0:
                print(f"    Processed {idx + 1}/{len(frames)} frames")

        # Convert to serializable format
        has_tracking = False  # Tracking not supported with pre-loaded frames
        predictions = convert_predictions_to_list(frame_results, has_tracking)
    else:
        # Run full inference pipeline
        print("  Running inference (this may take a while)...")
        results = run_inference(config, str(weights_path))

        # Convert to serializable format
        has_tracking = config.get("tracking", {}).get("enabled", False)
        predictions = convert_predictions_to_list(results["predictions"], has_tracking)

    # Cache results
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(predictions, f)
    print(f"  Cached to: {cache_path}")

    return predictions


def draw_detections(
    frame: np.ndarray,
    detections: list[list],
    color: tuple[int, int, int],
    label: str,
    scale_factor: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes and label on a scaled frame.

    Args:
        frame: Original frame (1280x720)
        detections: List of [x1, y1, x2, y2, conf, cls] lists
        color: BGR color for boxes
        label: Method name to display
        scale_factor: Resize factor (default 0.5 for 640x360)

    Returns:
        Scaled frame with annotations
    """
    # Make a copy to avoid modifying the original
    frame = frame.copy()

    # Scale frame with consistent interpolation
    h, w = frame.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Draw boxes (scale coordinates)
    for det in detections:
        x1, y1, x2, y2, conf, _ = det
        x1_s = int(x1 * scale_factor)
        y1_s = int(y1 * scale_factor)
        x2_s = int(x2 * scale_factor)
        y2_s = int(y2 * scale_factor)

        cv2.rectangle(scaled, (x1_s, y1_s), (x2_s, y2_s), color, 2)

        # Confidence label
        conf_text = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(scaled, (x1_s, y1_s - th - 4), (x1_s + tw + 4, y1_s), color, -1)
        cv2.putText(
            scaled,
            conf_text,
            (x1_s + 2, y1_s - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    # Method label (top-left with outline)
    cv2.putText(scaled, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(
        scaled, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1
    )

    # Detection count (bottom-left)
    count_text = f"Detections: {len(detections)}"
    cv2.putText(
        scaled,
        count_text,
        (10, new_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return scaled


def create_grid_frame(frames: list[np.ndarray]) -> np.ndarray:
    """
    Combine 4 frames into a 2x2 grid.

    Args:
        frames: List of 4 scaled frames (each 640x360)

    Returns:
        Combined grid (1280x720)
    """
    # Verify all frames have correct dimensions
    expected_shape = (360, 640, 3)
    for i, frame in enumerate(frames):
        if frame.shape != expected_shape:
            raise ValueError(
                f"Frame {i} has shape {frame.shape}, expected {expected_shape}"
            )

    top_row = np.hstack([frames[0], frames[1]])
    bottom_row = np.hstack([frames[2], frames[3]])
    grid = np.vstack([top_row, bottom_row])

    # Ensure output is correct size
    assert grid.shape == (720, 1280, 3), (
        f"Grid shape is {grid.shape}, expected (720, 1280, 3)"
    )

    return grid


def generate_comparison_video(
    video_id: str = "video_0",
    output_path: str = "outputs/videos/comparison_4up.mp4",
    cache_dir: str = "outputs/cache/predictions",
    fps: int = 30,
    force_rerun: bool = False,
    max_frames: int | None = None,
) -> None:
    """
    Generate 2x2 comparison video.

    Args:
        video_id: Video to process (video_0, video_1, video_2)
        output_path: Output MP4 path
        cache_dir: Directory for prediction cache
        fps: Video frame rate
        force_rerun: If True, rerun inference even if cached
        max_frames: Limit number of frames (for testing)
    """
    cache_path = Path(cache_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load frames (optimized for max_frames)
    frame_dir = f"data/train_images/{video_id}"
    print(f"\nLoading frames from: {frame_dir}")

    if max_frames:
        # Load only needed frames
        frame_paths = sorted(Path(frame_dir).glob("*.jpg"))[:max_frames]
        frames = [cv2.imread(str(p)) for p in frame_paths]
        print(f"Loaded {len(frames)} frames for testing")
    else:
        # Load all frames
        frames = load_frames(frame_dir)
        print(f"Loaded {len(frames)} frames")

    print(f"Processing {len(frames)} frames")

    # Load/run inference for each method
    all_predictions = {}
    for method in METHODS:
        print(f"\n[{method.name}]")
        all_predictions[method.experiment_id] = load_or_run_inference(
            method,
            video_id,
            cache_path,
            force_rerun,
            frames=frames if max_frames else None,
        )

    # Setup video writer with better codec settings
    # Try H.264 codec first (better quality, less flickering)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
    writer = cv2.VideoWriter(str(output), fourcc, fps, (1280, 720))

    # Check if writer opened successfully
    if not writer.isOpened():
        print("  Warning: H.264 codec not available, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output), fourcc, fps, (1280, 720))

    print("\nGenerating video...")
    # Process each frame
    for idx, frame in enumerate(frames):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  Frame {idx + 1}/{len(frames)}")

        grid_inputs = []
        for method in METHODS:
            preds = all_predictions[method.experiment_id]
            # Ensure we have predictions for this frame
            if idx < len(preds):
                frame_dets = preds[idx]
            else:
                frame_dets = []
                print(f"  Warning: No predictions for frame {idx} in {method.name}")

            annotated = draw_detections(frame, frame_dets, method.color, method.name)
            grid_inputs.append(annotated)

        grid_frame = create_grid_frame(grid_inputs)
        # Ensure frame is uint8 and contiguous
        grid_frame = np.ascontiguousarray(grid_frame, dtype=np.uint8)
        writer.write(grid_frame)

    writer.release()
    print(f"\n✓ Video saved to: {output}")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate 2x2 comparison video of COTS detection methods"
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default="video_0",
        help="Video to process (video_0, video_1, video_2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/videos/comparison_4up.mp4",
        help="Output video path",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument(
        "--no-cache", action="store_true", help="Force re-run inference (ignore cache)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Limit frames for testing"
    )
    args = parser.parse_args()

    generate_comparison_video(
        video_id=args.video_id,
        output_path=args.output,
        fps=args.fps,
        force_rerun=args.no_cache,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
