"""
Train a single ablation experiment.

Used by ablation_study.py to run individual experiments.
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--description", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--box", type=float, required=True)
    parser.add_argument("--hsv_h", type=float, required=True)
    parser.add_argument("--hsv_s", type=float, required=True)
    parser.add_argument("--hsv_v", type=float, required=True)
    parser.add_argument("--degrees", type=float, required=True)
    parser.add_argument("--mixup", type=float, required=True)

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Ablation Experiment: {args.name}")
    print(f"Description: {args.description}")
    print(f"{'='*80}\n")

    model = YOLO("yolo11n.pt")

    results = model.train(
        data=f"configs/dataset_fold_{args.fold}.yaml",
        epochs=args.epochs,
        imgsz=640,
        batch=16,
        device=args.device,
        project="runs/ablation",
        name=f"{args.name}_fold{args.fold}",
        patience=10,
        save=True,
        plots=True,
        val=True,
        amp=True,
        verbose=True,
        # Experiment parameters
        box=args.box,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        mixup=args.mixup,
    )

    print(f"\n{'-'*80}")
    print(f"Experiment: {args.name}")
    print(f"Description: {args.description}")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
    print(f"{'-'*80}\n")


if __name__ == "__main__":
    main()
