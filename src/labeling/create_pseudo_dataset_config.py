import argparse
import os
import yaml

def main():
    parser = argparse.ArgumentParser(description="Create YOLO dataset.yaml for pseudo-labeled data.")
    parser.add_argument("--dataset_name", type=str, default="pseudo_dataset", help="Name of the dataset.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset (containing images/ and labels/).")
    parser.add_argument("--class_name", type=str, default="object", help="Name of the class (e.g. starfish).")
    parser.add_argument("--output_path", type=str, default="configs/dataset_pseudo.yaml", help="Path to save the yaml config.")
    
    args = parser.parse_args()
    
    # Verify structure
    images_dir = os.path.join(args.dataset_root, "images")
    labels_dir = os.path.join(args.dataset_root, "labels")
    
    if not os.path.exists(images_dir):
        print(f"Warning: {images_dir} does not exist. Ensure your data structure is correct.")
        print(f"Expected: {args.dataset_root}/images and {args.dataset_root}/labels")
        
    config = {
        "path": os.path.abspath(args.dataset_root),
        "train": "images",
        "val": "images", # Use same for val if just training on all, or split manually
        "nc": 1,
        "names": {
            0: args.class_name
        }
    }
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
        
    print(f"Created dataset config at {args.output_path}")
    print("You can now train using:")
    print(f"yolo train data={args.output_path} model=yolo11n.pt epochs=10 ...")

if __name__ == "__main__":
    main()

