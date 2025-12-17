"""Quick script to check image dimensions"""
from PIL import Image
from pathlib import Path

# Check a sample image
sample_image = Path("data/train_images/video_0/0.jpg")
if sample_image.exists():
    img = Image.open(sample_image)
    print(f"Image dimensions: {img.size} (width x height)")
    print(f"Image mode: {img.mode}")
    print(f"Image format: {img.format}")
else:
    print("Sample image not found")
