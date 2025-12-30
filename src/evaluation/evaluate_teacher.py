import os
import argparse
import numpy as np
from tqdm import tqdm
import glob

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] normalized
    y = np.copy(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    """
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    lt = np.maximum(box1[:, None, :2], box2[:, :2])
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union

def load_labels(label_path):
    if not os.path.exists(label_path):
        return np.array([])
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            # class x y w h
            cls, x, y, w, h = map(float, parts)
            boxes.append([x, y, w, h])
            
    return np.array(boxes)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pseudo Labels against Ground Truth.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing Ground Truth YOLO labels.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing Predicted (Pseudo) YOLO labels.")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for a match.")
    
    args = parser.parse_args()
    
    # Get intersection of files
    pred_files = glob.glob(os.path.join(args.pred_dir, "*.txt"))
    pred_filenames = {os.path.basename(f) for f in pred_files}
    
    gt_files = glob.glob(os.path.join(args.gt_dir, "*.txt"))
    gt_filenames = {os.path.basename(f) for f in gt_files}
    
    common_files = pred_filenames.intersection(gt_filenames)
    
    if len(common_files) == 0:
        print("Error: No common filenames found between GT and Preds!")
        print(f"GT sample: {list(gt_filenames)[:3]}")
        print(f"Pred sample: {list(pred_filenames)[:3]}")
        return

    tp = 0 # True Positives
    fp = 0 # False Positives
    fn = 0 # False Negatives
    
    print(f"Evaluating {len(common_files)} common files...")
    
    for filename in tqdm(common_files):
        gt_path = os.path.join(args.gt_dir, filename)
        pred_path = os.path.join(args.pred_dir, filename)
        
        gt_boxes_xywh = load_labels(gt_path)
        pred_boxes_xywh = load_labels(pred_path)
        
        # Convert to xyxy for IoU
        if len(gt_boxes_xywh) > 0:
            gt_boxes = np.array([xywh2xyxy(b) for b in gt_boxes_xywh])
        else:
            gt_boxes = np.array([])
            
        if len(pred_boxes_xywh) > 0:
            pred_boxes = np.array([xywh2xyxy(b) for b in pred_boxes_xywh])
        else:
            pred_boxes = np.array([])
            
        # Evaluation Logic
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue
        elif len(gt_boxes) == 0:
            fp += len(pred_boxes)
            continue
        elif len(pred_boxes) == 0:
            fn += len(gt_boxes)
            continue
            
        # Match boxes
        iou_matrix = box_iou(pred_boxes, gt_boxes)
        
        # Greedy matching
        matched_gt = set()
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = iou_matrix[i, j]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= args.iou_thresh:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
                
        # Remaining GTs are False Negatives
        fn += len(gt_boxes) - len(matched_gt)
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nResults:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print("-" * 20)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    main()

