"""
Experiment Runner

Single entrypoint for running standardized experiments.
Loads experiment configs, runs training/inference/evaluation, and logs results.

Usage:
    uv run src/run_experiment.py --config configs/experiments/exp01_yolov5_baseline.yaml
    uv run src/run_experiment.py --config configs/experiments/exp02_yolov11_baseline.yaml
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.training.train_baseline import train_from_config
from src.inference.inference_utils import run_inference
from src.evaluation.evaluation_utils import evaluate_from_config


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_training(config: dict) -> dict:
    """
    Run model training based on config.

    Returns training artifacts (weights path, metrics, etc.)
    """
    if config.get("skip_training", False):
        print(f"[INFO] Skipping training for {config['experiment_id']}")
        weights_path = config.get("weights", "")
        if not Path(weights_path).exists():
            print(f"[WARNING] Weights not found: {weights_path}")
            print("[INFO] Please train the model first or provide correct weights path")
        return {"weights": weights_path}

    print(f"[INFO] Training {config['model']} for {config.get('epochs', 50)} epochs...")

    # Call training function with config
    results = train_from_config(config)

    return results


def run_evaluation(predictions: dict, config: dict) -> dict:
    """
    Evaluate predictions and compute metrics.

    Computes:
    - F2 score (primary metric)
    - mAP50, mAP50-95
    - Precision, Recall
    - Latency (ms/frame)

    Returns metrics dict.
    """
    eval_video_id = config.get("eval_video_id", f"video_{config.get('fold_id', 0)}")
    print(f"[INFO] Evaluating on {eval_video_id}...")

    # Call evaluation function with predictions and config
    metrics = evaluate_from_config(predictions, config)

    return metrics


def save_results(
    metrics: dict, config: dict, output_path: str = "outputs/metrics/results.csv"
):
    """
    Append experiment results to CSV.

    Follows standardized schema:
    experiment_id, fold_id, eval_video_id, model, inference, tracking,
    conf, iou, imgsz, f2, map50, recall, precision, ms_per_frame, seed, timestamp
    """
    # Extract values with defaults
    inference_config = config.get("inference", {})
    tracking_config = config.get("tracking", {})

    result = {
        "experiment_id": config.get("experiment_id", "unknown"),
        "fold_id": config.get("fold_id", 0),
        "eval_video_id": config.get(
            "eval_video_id", f"video_{config.get('fold_id', 0)}"
        ),
        "model": config.get("model", "yolo11n"),
        "inference": inference_config.get("mode", "standard"),
        "tracking": tracking_config.get("tracker", "bytetrack")
        if tracking_config.get("enabled", False)
        else "none",
        "conf": inference_config.get("conf", 0.25),
        "iou": inference_config.get("iou", 0.45),
        "imgsz": config.get("imgsz", 640),
        "f2": metrics.get("f2", 0.0),
        "map50": metrics.get("map50", 0.0),
        "recall": metrics.get("recall", 0.0),
        "precision": metrics.get("precision", 0.0),
        "ms_per_frame": metrics.get("ms_per_frame", 0.0),
        "seed": config.get("seed", 42),
        "timestamp": datetime.now().isoformat(),
    }

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Append to CSV
    file_exists = output_file.exists()

    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists or output_file.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(result)

    print(f"[SUCCESS] Results saved to {output_path}")
    print(
        f"  F2: {result['f2']:.4f}, mAP50: {result['map50']:.4f}, Recall: {result['recall']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run standardized experiment")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and print plan without executing",
    )
    args = parser.parse_args()

    # Load configuration
    print(f"[INFO] Loading config: {args.config}")
    config = load_config(args.config)

    print(f"\n{'=' * 60}")
    print(f"Experiment: {config['experiment_id']}")
    print(f"Description: {config['description']}")
    print(f"Model: {config['model']}")
    print(f"Inference: {config['inference']['mode']}")
    print(f"Tracking: {'enabled' if config['tracking']['enabled'] else 'disabled'}")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print("[DRY RUN] Skipping execution")
        return

    # Execution pipeline
    start_time = time.time()

    # 1. Training (or load existing weights)
    training_results = run_training(config)
    weights_path = training_results["weights"]

    # 2. Check for parameter sweeps
    inference_config = config.get("inference", {})
    conf_sweep = inference_config.get("conf_sweep", None)
    iou_sweep = inference_config.get("iou_sweep", None)

    if conf_sweep or iou_sweep:
        # Run sweep experiments
        print("\n[INFO] Running parameter sweep:")
        if conf_sweep:
            print(f"  conf_sweep: {conf_sweep}")
        if iou_sweep:
            print(f"  iou_sweep: {iou_sweep}")

        # Generate sweep configs
        sweep_configs = []
        if conf_sweep and iou_sweep:
            # Both sweeps - run cartesian product
            for conf_val in conf_sweep:
                for iou_val in iou_sweep:
                    sweep_configs.append({"conf": conf_val, "iou": iou_val})
        elif conf_sweep:
            # Only conf sweep
            for conf_val in conf_sweep:
                sweep_configs.append(
                    {"conf": conf_val, "iou": inference_config.get("iou", 0.45)}
                )
        elif iou_sweep:
            # Only iou sweep
            for iou_val in iou_sweep:
                sweep_configs.append(
                    {"conf": inference_config.get("conf", 0.25), "iou": iou_val}
                )

        # Run each sweep configuration
        for idx, sweep_params in enumerate(sweep_configs, 1):
            print(
                f"\n[INFO] Sweep {idx}/{len(sweep_configs)}: conf={sweep_params['conf']}, iou={sweep_params['iou']}"
            )

            # Create modified config for this sweep
            sweep_config = config.copy()
            sweep_config["inference"] = inference_config.copy()
            sweep_config["inference"]["conf"] = sweep_params["conf"]
            sweep_config["inference"]["iou"] = sweep_params["iou"]

            # 2. Inference
            predictions = run_inference(sweep_config, weights_path)

            # 3. Evaluation
            metrics = run_evaluation(predictions, sweep_config)

            # 4. Save results
            save_results(metrics, sweep_config)

    else:
        # Standard single-config experiment
        # 2. Inference
        predictions = run_inference(config, weights_path)

        # 3. Evaluation
        metrics = run_evaluation(predictions, config)

        # 4. Save results
        save_results(metrics, config)

    elapsed = time.time() - start_time
    print(f"\n[INFO] Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
