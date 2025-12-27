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


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_training(config: dict) -> dict:
    """
    Run model training based on config.

    Returns training artifacts (weights path, metrics, etc.)
    """
    if config.get('skip_training', False):
        print(f"[INFO] Skipping training for {config['experiment_id']}")
        return {'weights': config['weights']}

    print(f"[INFO] Training {config['model']} for {config['epochs']} epochs...")

    # TODO: Call existing training scripts
    # from training.train_baseline import train
    # results = train(
    #     model=config['model'],
    #     data=config['data'],
    #     epochs=config['epochs'],
    #     imgsz=config['imgsz'],
    #     batch=config['batch'],
    #     device=config['device'],
    #     ...
    # )

    raise NotImplementedError("Training integration pending")


def run_inference(config: dict, weights_path: str) -> dict:
    """
    Run inference based on config.

    Supports:
    - Standard inference
    - SAHI tiled inference
    - Confidence/IoU threshold sweeps
    - ByteTrack tracking

    Returns predictions and latency metrics.
    """
    print(f"[INFO] Running {config['inference']['mode']} inference...")

    # TODO: Call existing inference scripts
    # from inference.predict import predict
    # from inference.sahi_predict import sahi_predict (if SAHI)
    # predictions = predict(
    #     weights=weights_path,
    #     source=val_images,
    #     conf=config['inference']['conf'],
    #     iou=config['inference']['iou'],
    #     ...
    # )

    raise NotImplementedError("Inference integration pending")


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
    print(f"[INFO] Evaluating on {config['eval_video_id']}...")

    # TODO: Call existing evaluation scripts
    # from evaluation.f2_score import compute_f2
    # metrics = compute_f2(predictions, ground_truth)

    raise NotImplementedError("Evaluation integration pending")


def save_results(metrics: dict, config: dict, output_path: str = "outputs/metrics/results.csv"):
    """
    Append experiment results to CSV.

    Follows standardized schema:
    experiment_id, fold_id, eval_video_id, model, inference, tracking,
    conf, iou, imgsz, f2, map50, recall, precision, ms_per_frame, seed, timestamp
    """
    result = {
        'experiment_id': config['experiment_id'],
        'fold_id': config['fold_id'],
        'eval_video_id': config['eval_video_id'],
        'model': config['model'],
        'inference': config['inference']['mode'],
        'tracking': 'bytetrack' if config['tracking']['enabled'] else 'none',
        'conf': config['inference'].get('conf', 0.25),
        'iou': config['inference'].get('iou', 0.45),
        'imgsz': config.get('imgsz', 640),
        'f2': metrics['f2'],
        'map50': metrics['map50'],
        'recall': metrics['recall'],
        'precision': metrics['precision'],
        'ms_per_frame': metrics['ms_per_frame'],
        'seed': config.get('seed', 42),
        'timestamp': datetime.now().isoformat()
    }

    # Append to CSV
    output_file = Path(output_path)
    file_exists = output_file.exists()

    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists or output_file.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(result)

    print(f"[SUCCESS] Results saved to {output_path}")
    print(f"  F2: {result['f2']:.4f}, mAP50: {result['map50']:.4f}, Recall: {result['recall']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run standardized experiment")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment config YAML'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Load config and print plan without executing'
    )
    args = parser.parse_args()

    # Load configuration
    print(f"[INFO] Loading config: {args.config}")
    config = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"Experiment: {config['experiment_id']}")
    print(f"Description: {config['description']}")
    print(f"Model: {config['model']}")
    print(f"Inference: {config['inference']['mode']}")
    print(f"Tracking: {'enabled' if config['tracking']['enabled'] else 'disabled'}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("[DRY RUN] Skipping execution")
        return

    # Execution pipeline
    start_time = time.time()

    # 1. Training (or load existing weights)
    training_results = run_training(config)
    weights_path = training_results['weights']

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
