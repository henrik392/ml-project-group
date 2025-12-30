import os
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
import json

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def convert_to_yolo(bbox, image_width, image_height):
    """
    Convert [x1, y1, x2, y2] to [x_center, y_center, width, height] normalized.
    """
    x1, y1, x2, y2 = bbox
    
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w / 2
    y_center = y1 + h / 2
    
    x_center /= image_width
    y_center /= image_height
    w /= image_width
    h /= image_height
    
    return x_center, y_center, w, h

def main():
    parser = argparse.ArgumentParser(description="Use Florence-2 to generate bounding boxes for images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save YOLO format labels.")
    parser.add_argument("--model_id", type=str, default="microsoft/Florence-2-large", help="HuggingFace model ID.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for specific object detection (e.g. 'crown of thorns starfish'). If not provided, runs generic OD.")
    parser.add_argument("--class_id", type=int, default=0, help="Class ID to use for YOLO format (default: 0).")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Confidence threshold (Florence-2 doesn't output scores easily, but keeping arg for future).")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu, mps). If not specified, auto-detects.")
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.device:
        device = args.device
    else:
        device = get_device()
    
    print(f"Loading model {args.model_id} on {device}...")
    if device == "cpu" and torch.cuda.is_available():
        print("Warning: CUDA is available but using CPU. Specify --device cuda to force GPU.")
    elif device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    # Load model and processor
    # Use float16 for GPU memory efficiency if on CUDA
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager"  # Avoid SDPA compatibility issues
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Determine task
    if args.prompt:
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        text_input = args.prompt
        print(f"Running '{task_prompt}' with text '{text_input}'")
    else:
        task_prompt = '<OD>'
        text_input = None
        print(f"Running '{task_prompt}' (Generic Object Detection)")
        
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {args.input_dir}")
    
    for filename in tqdm(image_files):
        image_path = os.path.join(args.input_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Could not open {filename}: {e}")
            continue
            
        # Prepare inputs
        if text_input:
             inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, dtype)
             # Note: For CAPTION_TO_PHRASE_GROUNDING, the prompt format might need to be explicitly constructed 
             # if processor doesn't handle it automatically with 'text' arg.
             # Standard usage: text = task_prompt + text_input
             inputs = processor(text=task_prompt + text_input, images=image, return_tensors="pt").to(device, dtype)
        else:
             inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, dtype)
        
        # Generate
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
            use_cache=False  # Disable KV cache to avoid incompatible past_key_values with eager attention
        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Post-process
        # Note: If we passed text_input combined with task_prompt, we should use that combined string or just the task for parsing?
        # processor.post_process_generation expects the same task string usually.
        # But for CAPTION_TO_PHRASE_GROUNDING, let's see.
        
        parse_task = task_prompt 
        
        results = processor.post_process_generation(
            generated_text, 
            task=parse_task, 
            image_size=(image.width, image.height)
        )
        
        # Extract boxes
        # results is like {'<OD>': {'bboxes': ..., 'labels': ...}} or {'<CAPTION_TO_PHRASE_GROUNDING>': ...}
        
        if task_prompt in results:
            data = results[task_prompt]
            bboxes = data.get('bboxes', [])
            labels = data.get('labels', [])
            
            if len(labels) > 0:
                print(f"Detected in {filename}: {labels}")
            
            # Write to label file
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(args.output_dir, label_filename)
            
            with open(label_path, "w") as f:
                for i, bbox in enumerate(bboxes):
                    # Filter by label if needed? 
                    # If prompt was provided, labels match the prompt usually.
                    # If generic OD, labels are classes.
                    
                    # For this pipeline, we assume we map everything found to the single class_id provided
                    # OR if the user wants to keep class names, we'd need a class map.
                    # The user asked to create bounding boxes (implied: for the object of interest).
                    
                    x_c, y_c, w, h = convert_to_yolo(bbox, image.width, image.height)
                    
                    # Basic validation
                    # Filter out boxes that are too large (likely hallucinations/entire image)
                    if w > 0.9 or h > 0.9:
                        continue
                        
                    if w > 0 and h > 0:
                        f.write(f"{args.class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        else:
            print(f"Warning: Task {task_prompt} not found in results for {filename}")

if __name__ == "__main__":
    main()

