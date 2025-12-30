import os
import argparse
import cv2
import matplotlib.pyplot as plt
import random

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO labels on images.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory containing YOLO labels.")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to visualize.")
    parser.add_argument("--shuffle", action="store_true", help="Randomly select images.")
    
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
        label_path = os.path.join(args.label_dir, label_filename)
        
        if not os.path.exists(label_path):
            continue
            
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Read labels
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            cls, x_c, y_c, box_w, box_h = map(float, parts)
            
            # Convert to x1, y1, x2, y2
            x1 = int((x_c - box_w / 2) * w)
            y1 = int((y_c - box_h / 2) * h)
            x2 = int((x_c + box_w / 2) * w)
            y2 = int((y_c + box_h / 2) * h)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"Class {int(cls)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(filename)
        plt.show()
        
        count += 1

if __name__ == "__main__":
    main()

