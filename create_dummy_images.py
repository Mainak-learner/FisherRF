import os
import json
from PIL import Image

# Path to your visualization folder
vis_path = "/content/FisherRF/my_vis"

# Load transforms.json
with open(os.path.join(vis_path, "transforms.json"), 'r') as f:
    transforms = json.load(f)

# Create dummy_images folder
dummy_img_dir = os.path.join(vis_path, "dummy_images")
os.makedirs(dummy_img_dir, exist_ok=True)

# Create black images
for idx, frame in enumerate(transforms["frames"]):
    img = Image.new("RGB", (512, 512), (0, 0, 0))  # Black 512x512
    img.save(os.path.join(dummy_img_dir, f"{idx:04d}.png"))

print(f"Generated {len(transforms['frames'])} dummy images!")