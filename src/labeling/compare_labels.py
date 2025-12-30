import os
import argparse
import cv2
import matplotlib.pyplot as plt
import random

def load_boxes(label_path, width, height):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        
        # YOLO format: class x_center y_center w h (normalized)
        cls, x_c, y_c, box_w, box_h = map(float, parts)
        
        # Convert to pixels: x1, y1, x2, y2
        x1 = int((x_c - box_w / 2) * width)
        y1 = int((y_c - box_h / 2) * height)
        x2 = int((x_c + box_w / 2) * width)
        y2 = int((y_c + box_h / 2) * height)
        
        boxes.append((x1, y1, x2, y2))
    return boxes

def main():
    parser = argparse.ArgumentParser(description="Compare Ground Truth vs Predicted Labels.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing Ground Truth YOLO labels.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing Predicted YOLO labels.")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to visualize.")
    parser.add_argument("--shuffle", action="store_true", help="Randomly select images.")
    parser.add_argument("--only_diff", action="store_true", help="Only show images where GT and Pred differ (or are non-empty).")
    
    args = parser.parse_args()
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in os.listdir(args.image_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    
    if args.shuffle:
        random.shuffle(image_files)
    
    count = 0
    for filename in image_files:
        if count >= args.num_images:
            break
            
        image_path = os.path.join(args.image_dir, filename)
        label_filename = os.path.splitext(filename)[0] + ".txt"
        gt_path = os.path.join(args.gt_dir, label_filename)
        pred_path = os.path.join(args.pred_dir, label_filename)
        
        # Skip check: if we only want interesting images
        if args.only_diff:
            has_gt = os.path.exists(gt_path) and os.path.getsize(gt_path) > 0
            has_pred = os.path.exists(pred_path) and os.path.getsize(pred_path) > 0
            if not has_gt and not has_pred:
                continue
            
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Load boxes
        gt_boxes = load_boxes(gt_path, w, h)
        pred_boxes = load_boxes(pred_path, w, h)
        
        # Draw GT (Green)
        for box in gt_boxes:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img, "GT", (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Draw Pred (Red)
        for box in pred_boxes:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(img, "Pred", (box[0], box[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        # Plot
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        
        title = f"{filename}\nGreen=GT ({len(gt_boxes)}), Red=Pred ({len(pred_boxes)})"
        plt.title(title)
        plt.show()
        
        count += 1

if __name__ == "__main__":
    main()

